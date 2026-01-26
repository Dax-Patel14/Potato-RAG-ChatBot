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

    # Unified RAG processing - LLM intelligently handles all query types
    with st.spinner("Thinking..."):
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
                    st.markdown("---")
                    with st.expander(f"📚 View Sources ({len(source_documents)} found)"):
                        for i, doc in enumerate(source_documents):
                            source_name = doc.metadata.get('source', 'Unknown Source')
                            st.markdown(f"**Source {i+1}:** `{source_name}`")
                            
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