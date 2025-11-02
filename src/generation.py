import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

def create_conversational_chain(retriever):
    """
    Creates a conversational RAG chain with memory, retriever, and LLM.
    """
    print("Initializing LLM and memory for conversational chain...")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    # Initialize a memory object to store chat history
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True,
        output_key='answer'  
    )
    
    # Create the ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        return_generated_question=True,
        # verbose = True
    )
    
    print("Conversational RAG chain created successfully.")
    return chain