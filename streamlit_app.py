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
try:
    from src.query_processor import QueryProcessor, ContextFilter
    ENHANCED_PROCESSING = True
except ImportError:
    print("Enhanced processing not available, using basic mode")
    ENHANCED_PROCESSING = False

# --- Page Setup ---
st.set_page_config(page_title="Aloo Sahayak 🥔", layout="wide")
st.title("💬 Aloo Sahayak: Your Potato Disease Assistant")
st.caption("Ask me about potato diseases based on the provided documents!")

# Initialize chat history BEFORE using it
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat_history = []

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # System status
    st.subheader("📊 System Status")
    if ENHANCED_PROCESSING:
        st.success("✅ Enhanced Processing Active")
        st.info("🔍 Advanced query processing and context filtering enabled")
    else:
        st.warning("⚠️ Basic Mode")
        st.info("💡 Install additional dependencies for enhanced features")
    
    # Statistics
    if st.session_state.chat_history:
        st.metric("Conversations", len(st.session_state.chat_history))
    
    # Clear conversation
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    
    # Help section
    st.subheader("💡 Tips for Better Results")
    st.markdown("""
    - **Be specific**: Mention disease names, symptoms, or plant parts
    - **Use context**: Refer to previous messages in follow-up questions  
    - **Ask about**: Symptoms, causes, treatments, prevention, management
    - **Examples**: 
      - "What are the symptoms of late blight?"
      - "How to prevent blackleg in potatoes?"
      - "Treatment for ring rot disease"
    """)

# --- Initialization ---
@st.cache_resource
def load_chain():
    retriever = load_retriever_from_disk()
    chain = create_conversational_chain(retriever)
    return chain

@st.cache_resource 
def load_processors():
    if ENHANCED_PROCESSING:
        query_processor = QueryProcessor()
        context_filter = ContextFilter()
        return query_processor, context_filter
    return None, None

qa_chain = load_chain()
query_processor, context_filter = load_processors()

# Initialize chat history (if not already done above)
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat_history = []

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
        # Enhanced RAG logic with improved processing
        with st.spinner("Analyzing your question and searching knowledge base..."):
            try:
                # Enhanced query processing if available
                if ENHANCED_PROCESSING and query_processor:
                    processed_query = query_processor.preprocess_question(
                        user_query, 
                        st.session_state.chat_history
                    )
                    query_to_use = processed_query['primary_query']
                    
                    # Show processed query in debug mode (optional)
                    if st.sidebar.checkbox("Show Query Processing", value=False):
                        st.sidebar.write("**Original:**", processed_query['original_question'])
                        st.sidebar.write("**Enhanced:**", processed_query['enhanced_question'])
                else:
                    query_to_use = user_query
                
                # Invoke the chain with the processed query
                result = qa_chain.invoke({
                    "question": query_to_use,
                    "chat_history": st.session_state.chat_history
                })
                
                ai_response = result["answer"]
                source_documents = result.get("source_documents", [])
                
                # Enhanced context filtering if available
                if ENHANCED_PROCESSING and context_filter and source_documents:
                    filtered_docs = context_filter.filter_contexts(
                        source_documents, 
                        user_query, 
                        max_contexts=6
                    )
                    source_documents = filtered_docs

                # Add AI response to display state
                st.session_state.messages.append(("assistant", ai_response))
                with st.chat_message("assistant"):
                    st.markdown(ai_response)
                    
                    # Enhanced source display
                    if source_documents:
                        with st.expander(f"📚 View Sources ({len(source_documents)} found)"):
                            for i, doc in enumerate(source_documents):
                                source_name = doc.metadata.get('source', 'Unknown Source')
                                doc_type = doc.metadata.get('type', 'text')
                                
                                # Different display for different document types
                                if doc_type == 'image_description':
                                    st.markdown(f"**📷 Image Source {i+1}:** `{source_name}`")
                                    st.markdown(f"*Image: {doc.metadata.get('image_name', 'N/A')}*")
                                else:
                                    st.markdown(f"**📄 Source {i+1}:** `{source_name}`")
                                
                                content_preview = doc.page_content[:300].replace('\n', ' ')
                                if len(doc.page_content) > 300:
                                    content_preview += "..."
                                st.markdown(f"> {content_preview}")
                                
                                if i < len(source_documents) - 1:
                                    st.divider()
                    else:
                        st.info("💡 No specific sources found. Response based on general knowledge.")

                # Add this Q&A to the RAG chain's memory
                st.session_state.chat_history.append((user_query, ai_response))
                
            except Exception as e:
                st.error(f"Sorry, I encountered an error: {str(e)}")
                st.info("Please try rephrasing your question or contact support if the issue persists.")