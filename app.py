import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains.retrieval_qa.base import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


st.set_page_config(page_title="GenAI RAG App")
st.title("📄 GenAI RAG App (Open Source)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    st.info("Processing document...")

    # Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector DB
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    # Load local HuggingFace model
    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    st.success("Document Ready! Ask questions 👇")

    query = st.text_input("Ask a question")

    if query:
        response = qa_chain.run(query)
        st.write("### Answer:")
        st.write(response)