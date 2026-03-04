# GenAI RAG App

This project demonstrates a **Retrieval-Augmented Generation (RAG) application** that allows users to upload a PDF document and ask questions about its content. The system retrieves relevant information from the document and generates answers using an open-source language model.

---

# Features

* Upload PDF documents
* Automatic text extraction
* Text chunking and embedding generation
* Vector search using FAISS
* Question answering using a language model
* Interactive UI built with Streamlit

---

# Tech Stack

* Python
* Streamlit
* LangChain
* HuggingFace Transformers
* FAISS Vector Database
* Sentence Transformers

---

# Challenges Faced

### 1. LangChain Import Issues

Initially, I faced errors while importing `RetrievalQA` from LangChain because the newer versions of LangChain changed the module structure.

**Solution:**
I resolved this by using the correct import path and installing compatible versions of LangChain packages.

---

### 2. Dependency Conflicts

While installing dependencies, I encountered conflicts between libraries such as `transformers`, `tokenizers`, and `scikit-learn`.

**Solution:**
I fixed this by specifying stable versions in the `requirements.txt` file and reinstalling the packages in a clean virtual environment.

---

### 3. HuggingFace API Token Requirement

At first, I used `HuggingFaceHub`, which required an API token for model access.

**Solution:**
To avoid dependency on API keys, I switched to using `HuggingFacePipeline` with locally loaded models like **Flan-T5**, which works directly on the CPU.

---

### 4. PDF Processing and Text Chunking

Handling large PDF documents required splitting the text into smaller chunks to ensure efficient retrieval.

**Solution:**
I used **RecursiveCharacterTextSplitter** to divide the document into manageable chunks before generating embeddings.

---

### 5. Deployment Preparation

Preparing the project for deployment required organizing the project files and ensuring all dependencies were properly listed.

**Solution:**
I structured the project with `app.py`, `requirements.txt`, and `README.md`, and tested the application locally before deployment.

---

# How to Run the Project

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

---

# Future Improvements

* Support for multiple PDF documents
* Chat history memory
* Better UI for conversation-style interaction
* Deployment with GPU support for faster responses
