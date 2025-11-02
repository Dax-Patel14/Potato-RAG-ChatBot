# --- Python Path Setup ---
import sys
import os
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Setup ---

import streamlit as st
from src.retrieval import load_retriever_from_disk
from src.generation import create_conversational_chain

# --- Page Setup ---
st.set_page_config(page_title="Aloo Sahayak 🥔", layout="wide")
st.title("💬 Aloo Sahayak: Your Potato Disease Assistant")
st.caption("Ask me about potato diseases based on the provided documents!")

# --- Initialization ---
@st.cache_resource
def load_chain():
    retriever = load_retriever_from_disk()
    chain = create_conversational_chain(retriever)
    return chain

qa_chain = load_chain()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat_history = [] # For RAG chain memory

# --- NEW: Intent Classifier ---
def classify_intent(query):
    """
    Classifies the user's intent as 'rag_question' or 'chit_chat'.
    """
    query_lower = query.lower().strip()
    
    # Simple keywords for non-RAG chat
    chit_chat_keywords = [
        'hello', 'hi', 'hey', 'good morning', 'good evening',
        'thanks', 'thank you', 'thx', 'appreciate it',
        'bye', 'goodbye', 'see you'
    ]
    
    # Check if any keyword is in the query
    if any(keyword in query_lower for keyword in chit_chat_keywords):
        return "chit_chat"
    
    return "rag_question"

# --- NEW: Chit-Chat Response Generator ---
def get_chit_chat_response(query):
    """
    Returns a pre-defined response for chit-chat.
    """
    query_lower = query.lower().strip()
    
    if any(keyword in query_lower for keyword in ['hello', 'hi', 'hey']):
        return "Hello! I'm Aloo Sahayak. How can I help you with potato diseases today?"
    if any(keyword in query_lower for keyword in ['thanks', 'thank you']):
        return "You're welcome! Do you have any other questions?"
    if any(keyword in query_lower for keyword in ['bye', 'goodbye']):
        return "Goodbye! Have a great day."
        
    return "I'm sorry, I can only assist with questions about potato diseases."

# --- Display existing messages ---
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

# --- Handle User Input ---
user_query = st.chat_input("Ask about potato diseases...")

if user_query:
    # Add user message to display state
    st.session_state.messages.append(("user", user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    # --- UPDATED: Intent Routing Logic ---
    intent = classify_intent(user_query)

    if intent == "chit_chat":
        # Get and display the pre-defined chit-chat response
        ai_response = get_chit_chat_response(user_query)
        st.session_state.messages.append(("assistant", ai_response))
        with st.chat_message("assistant"):
            st.markdown(ai_response)
        # NOTE: We do NOT add this to the RAG chain's memory (`st.session_state.chat_history`)
        # This prevents polluting the memory with irrelevant context.
            
    elif intent == "rag_question":
        # This is the original RAG logic
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({
                "question": user_query,
                "chat_history": st.session_state.chat_history
            })
            ai_response = result["answer"]
            source_documents = result.get("source_documents", [])

            # Add AI response to display state
            st.session_state.messages.append(("assistant", ai_response))
            with st.chat_message("assistant"):
                st.markdown(ai_response)
                
                # Add Expander for Sources
                if source_documents:
                    with st.expander("View Sources"):
                        for i, doc in enumerate(source_documents):
                            source_name = doc.metadata.get('source', 'Unknown Source')
                            st.markdown(f"**Source {i+1}:** `{source_name}`")
                            content_preview = doc.page_content[:250].replace('\n', ' ') + "..."
                            st.markdown(f"> {content_preview}")
                            st.divider()

            # Add this Q&A to the RAG chain's memory
            st.session_state.chat_history.append((user_query, ai_response))