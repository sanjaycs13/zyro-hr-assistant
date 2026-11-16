import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Page setup matching the clean centered chat UI
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 HR Policy Assistant")
st.caption("Ask me anything about the company HR policy document.")

CORPUS_PATH = "./zyro-dynamics-hr-corpus"

REFUSAL_MESSAGE = (
    "I can only answer questions about Zyro Dynamics' internal HR policies. "
    "This topic is not covered in the available HR policy documents."
)

@st.cache_resource(show_spinner="Loading HR Policy PDFs and building FAISS index...")
def load_retriever():
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()
    for d in documents:
        d.metadata["source_doc"] = os.path.basename(d.metadata.get("source", "HR Policy.pdf"))
        d.metadata["page_num"] = d.metadata.get("page", 0) + 1

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.85}
    )

retriever = load_retriever()

# Read keys safely from Streamlit Secrets (so GitHub Push Protection never blocks your push)
GROQ_KEY_1 = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
GROQ_KEY_2 = st.secrets.get("GROQ_API_KEY_2", os.environ.get("GROQ_API_KEY_2", GROQ_KEY_1))

llm_model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_tokens=700,
    api_key=GROQ_KEY_1
).with_fallbacks([
    ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.0, max_tokens=700, api_key=GROQ_KEY_2),
    ChatGroq(model="llama-3.1-8b-instant", temperature=0.0, max_tokens=700, api_key=GROQ_KEY_2),
])

GUARDIAL_PROMPT = ChatPromptTemplate.from_template("""
You are a strict scope classifier for Zyro Dynamics' internal HR Help Desk.
IN_SCOPE topics (ONLY internal employee HR policies):
- Greetings ("hi", "hello", "who are you")
- Company overview, employee handbook, onboarding, probation, resignation/notice period
- Leave policy (earned, sick, casual, maternity, paternity) and Work From Home (WFH)
- Salary credit date, salary grades/ranges, medical insurance, annual performance review (APR), PIP
- Code of conduct, gifts, IT & data security (passwords/devices), POSH/ICC complaints, travel & expense reimbursements

OUT_OF_SCOPE topics (MUST be refused):
- Applying for a job / external hiring / careers
- ESOP / stock options (if not in internal HR policy)
- Company revenue, financials, or product comparisons (e.g., ZyroCRM vs Salesforce)
- Other companies (e.g., Zoho) or general trivia (sports, IPL, capitals, coding)

Reply with ONLY one word: IN_SCOPE or OUT_OF_SCOPE.

Question: {question}
""")

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are Zyro Dynamics' friendly HR Help Desk Assistant. Answer using ONLY the Context below.\n"
        "1. If the user simply greets you (e.g. 'hi who are you'), introduce yourself warmly as the Zyro Dynamics HR Policy Assistant.\n"
        "2. Otherwise, be direct and concise (1 to 4 plain-text sentences, no bullet points, no * or ** bold markdown).\n"
        "3. If a policy rule, allowance, or eligibility depends on employee grade/level (e.g., L1-L3, L4-L6) and the user didn't specify their grade, list the values for all grades.\n"
        "4. Include every exact number, days, percentage, grade, amount, and timeline stated in the policy.\n"
        "5. Only if the answer is completely absent from the Context, reply with:\n"
        f"{REFUSAL_MESSAGE}"
    )),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
])

# Keep chat history on screen just like a real chat app
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources Cited"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['doc']} — Page {s['page']}**")
                    st.caption(s["text"])

# Chat input box at the bottom
if question := st.chat_input("Ask a question about HR policy..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            guardial_chain = GUARDIAL_PROMPT | llm_model | StrOutputParser()
            verdict = guardial_chain.invoke({"question": question}).strip().upper()

            if "OUT_OF_SCOPE" in verdict:
                answer = REFUSAL_MESSAGE
                sources_list = []
                st.write(answer)
            else:
                docs = retriever.invoke(question)
                context = "\n\n".join(
                    f"[Document: {d.metadata.get('source_doc')} | Page: {d.metadata.get('page_num')}]\n{d.page_content}"
                    for d in docs
                )
                rag_chain = RAG_PROMPT | llm_model | StrOutputParser()
                raw_answer = rag_chain.invoke({"context": context, "question": question})
                answer = " ".join(raw_answer.replace("**", "").replace("*", "").split())

                sources_list = [
                    {
                        "doc": d.metadata.get("source_doc"),
                        "page": d.metadata.get("page_num"),
                        "text": d.page_content[:300] + "..."
                    }
                    for d in docs
                ]

                st.write(answer)
                if sources_list:
                    with st.expander("📄 Sources Cited"):
                        for s in sources_list:
                            st.markdown(f"**{s['doc']} — Page {s['page']}**")
                            st.caption(s["text"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources_list
    })