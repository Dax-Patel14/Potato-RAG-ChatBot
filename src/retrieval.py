# File: src/2_retrieval.py
from langchain_openai import OpenAIEmbeddings 
from langchain_community.vectorstores import FAISS

FAISS_INDEX_PATH = "faiss_index"

def load_retriever_from_disk():
    # Use the same OpenAI embedding model to load the index
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small") 
    
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    return vector_store.as_retriever(search_kwargs={"k": 3})

def retrieve_docs(retriever, question: str):
    return retriever.invoke(question)