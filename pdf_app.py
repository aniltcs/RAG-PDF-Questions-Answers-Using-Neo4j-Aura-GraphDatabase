import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores.neo4j_vector import Neo4jVector


# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN


# Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ----------------------------
# LLM: Groq Llama 3.1
# ----------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# ----------------------------
# Embedding model
# ----------------------------
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ----------------------------
# Prompt Template
# ----------------------------
prompt_template = """
You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer concisely and clearly:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📘 Neo4j Vector RAG — PDF Document Q&A (Groq + Llama3)")

if "neo4j_vector" not in st.session_state:
    st.session_state.neo4j_vector = None


# ----------------------------
# Create Vector Index in Neo4j
# ----------------------------
def create_neo4j_vector_index():

    loader = PyPDFDirectoryLoader("research_papers")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    vector_store = Neo4jVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name="pdf_index",
        node_label="PDFChunk",
        text_node_property="text",
        embedding_node_property="embedding",
    )

    st.session_state.neo4j_vector = vector_store


# ----------------------------
# Button: Create Embeddings
# ----------------------------
if st.button("📥 Create Neo4j Vector Index from PDFs"):
    with st.spinner("Processing PDFs and generating embeddings..."):
        create_neo4j_vector_index()
    st.success("Neo4j vector index created successfully!")


# ----------------------------
# User Input
# ----------------------------
user_query = st.text_input("🔎 Ask a question from the PDF knowledge base:")

if user_query and st.session_state.neo4j_vector:

    retriever = st.session_state.neo4j_vector.as_retriever()

    # Retrieved docs
    relevant_docs = retriever.invoke(user_query)

    # Build RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    start = time.process_time()
    answer = rag_chain.invoke(user_query)
    elapsed = time.process_time() - start

    st.subheader("🧠 Answer")
    st.write(answer)

    st.caption(f"⏱️ Response time: {elapsed:.2f} sec")

    # Show relevant chunks
    with st.expander("📄 Retrieved Chunks"):
        for doc in relevant_docs:
            st.write(doc.page_content)
            st.write("---")

elif user_query:
    st.warning("⚠️ Please create the vector index first by clicking the button above.")
