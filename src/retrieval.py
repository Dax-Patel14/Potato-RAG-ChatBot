from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Point this to the new multimodal index directory
FAISS_INDEX_PATH = "faiss_index_multimodal" # <-- Update this line

def load_retriever_from_disk():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    return vector_store.as_retriever(search_kwargs={"k": 5}) # Increased k for richer context

def retrieve_docs(retriever, question: str):
    return retriever.invoke(question)