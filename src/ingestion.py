# File: src/1_ingestion.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
PDF_PATH = os.path.join("data", "Fungal_diseases_of_potato.pdf")
FAISS_INDEX_PATH = "faiss_index"

def load_pdf(file_path: str) -> list[Document]:
    loader = PyPDFLoader(file_path)
    return loader.load()

def split_document(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)

def embed_and_store_faiss(chunks: list[Document]):
    print("Initializing OpenAI embedding model...")
    # Use OpenAI's state-of-the-art embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("Creating FAISS index with OpenAI embeddings...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    print(f"FAISS index created and saved at {FAISS_INDEX_PATH}")

def run_ingestion():
    print("Starting ingestion pipeline...")
    documents = load_pdf(PDF_PATH)
    chunks = split_document(documents)
    embed_and_store_faiss(chunks)
    print("--- Ingestion Complete! ---")

if __name__ == "__main__":
    run_ingestion()