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

# --- Import Multimodal Generator ---
try:
    from src.multimodal_generation import create_multimodal_generator
    MULTIMODAL_AVAILABLE = True
except ImportError:
    print("Multimodal module not found. Running in text-only mode.")
    MULTIMODAL_AVAILABLE = False

try:
    from src.query_processor import QueryProcessor, ContextFilter
    ENHANCED_PROCESSING = True
except ImportError:
    print("Enhanced processing not available, using basic mode")
    ENHANCED_PROCESSING = False

# --- Page Setup ---
st.set_page_config(page_title="Aloo Sahayak", layout="wide")
st.title("🥔 Aloo Sahayak: Your Potato Disease Assistant")
st.caption("Ask me about potato diseases based on the provided documents!")

# Initialize response language preference
if "response_language" not in st.session_state:
    st.session_state.response_language = "English"

# Initialize multi-chat state
import datetime
import uuid
import json
from pathlib import Path

# Path to save chat history
CHAT_HISTORY_FILE = Path("chat_history.json")

def save_chats_to_disk():
    """Save all chats to disk"""
    try:
        # Convert datetime objects to strings for JSON serialization
        chats_to_save = {}
        for chat_id, chat_data in st.session_state.chats.items():
            chats_to_save[chat_id] = {
                "name": chat_data["name"],
                "messages": chat_data["messages"],
                "chat_history": chat_data["chat_history"],
                "created_at": chat_data["created_at"].isoformat()
            }
        
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                "chats": chats_to_save,
                "active_chat_id": st.session_state.active_chat_id
            }, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving chats: {e}")

def load_chats_from_disk():
    """Load all chats from disk"""
    try:
        if CHAT_HISTORY_FILE.exists():
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert string dates back to datetime objects
            chats = {}
            for chat_id, chat_data in data["chats"].items():
                chats[chat_id] = {
                    "name": chat_data["name"],
                    "messages": chat_data["messages"],
                    "chat_history": chat_data["chat_history"],
                    "created_at": datetime.datetime.fromisoformat(chat_data["created_at"])
                }
            
            return chats, data.get("active_chat_id")
    except Exception as e:
        print(f"Error loading chats: {e}")
    
    return None, None

if "chats" not in st.session_state:
    # Try to load from disk first
    loaded_chats, loaded_active_id = load_chats_from_disk()
    
    if loaded_chats:
        # Restore from saved state
        st.session_state.chats = loaded_chats
        st.session_state.active_chat_id = loaded_active_id
    else:
        # Create default chat if no saved state
        default_chat_id = str(uuid.uuid4())
        st.session_state.chats = {
            default_chat_id: {
                "name": "New Chat",
                "messages": [],
                "chat_history": [],
                "created_at": datetime.datetime.now()
            }
        }
        st.session_state.active_chat_id = default_chat_id
        save_chats_to_disk()

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = list(st.session_state.chats.keys())[0]

# Helper functions for chat management
def create_new_chat():
    """Create a new chat session"""
    new_chat_id = str(uuid.uuid4())
    st.session_state.chats[new_chat_id] = {
        "name": "New Chat",
        "messages": [],
        "chat_history": [],
        "created_at": datetime.datetime.now()
    }
    st.session_state.active_chat_id = new_chat_id
    save_chats_to_disk()

def delete_chat(chat_id):
    """Delete a specific chat"""
    if len(st.session_state.chats) > 1 and chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
        # Switch to another chat
        st.session_state.active_chat_id = list(st.session_state.chats.keys())[0]
        save_chats_to_disk()

def auto_rename_chat(chat_id, first_message):
    """Auto-rename chat based on first user message"""
    if st.session_state.chats[chat_id]["name"] == "New Chat":
        # Use first 50 chars of first message as chat name
        name = first_message[:50] + ("..." if len(first_message) > 50 else "")
        st.session_state.chats[chat_id]["name"] = name
        save_chats_to_disk()

def rename_chat(chat_id, new_name):
    """Manually rename a chat"""
    if chat_id in st.session_state.chats and new_name.strip():
        st.session_state.chats[chat_id]["name"] = new_name.strip()
        save_chats_to_disk()

# Get current active chat
active_chat = st.session_state.chats[st.session_state.active_chat_id]

# Sidebar for chat management
with st.sidebar:
    st.header("💬 Conversations")
    
    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        create_new_chat()
        st.rerun()
    
    st.divider()
    
    # List all chats
    st.subheader("🕰️ Chat History")
    
    # Sort chats by creation time (newest first)
    sorted_chats = sorted(
        st.session_state.chats.items(),
        key=lambda x: x[1]["created_at"],
        reverse=True
    )
    
    for chat_id, chat_data in sorted_chats:
        col1, col2 = st.columns([5, 1])
        
        with col1:
            # Chat selection button
            is_active = chat_id == st.session_state.active_chat_id
            
            # Show chat name with message count
            msg_count = len(chat_data["messages"]) // 2  # Divide by 2 (user + assistant)
            chat_label = f"{'🔹 ' if is_active else ''}{chat_data['name']}"
            if msg_count > 0:
                chat_label += f" ({msg_count})"
            
            if st.button(chat_label, key=f"chat_{chat_id}", use_container_width=True):
                if not is_active:
                    st.session_state.active_chat_id = chat_id
                    save_chats_to_disk()
                    st.rerun()
        
        with col2:
            # Three-dot menu for rename and delete
            with st.popover("⋮", use_container_width=True):
                st.markdown(f"**{chat_data['name'][:30]}...**" if len(chat_data['name']) > 30 else f"**{chat_data['name']}**")
                st.divider()
                
                # Rename option
                new_name = st.text_input(
                    "Rename chat",
                    value=chat_data['name'],
                    key=f"rename_{chat_id}",
                    label_visibility="collapsed",
                    placeholder="Enter new name..."
                )
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("✏️ Rename", key=f"rename_btn_{chat_id}", use_container_width=True):
                        if new_name and new_name != chat_data['name']:
                            rename_chat(chat_id, new_name)
                            st.rerun()
                
                # Delete option (only if more than 1 chat exists)
                if len(st.session_state.chats) > 1:
                    st.divider()
                    if st.button("🗑️ Delete", key=f"del_{chat_id}", use_container_width=True, type="secondary"):
                        delete_chat(chat_id)
                        st.rerun()
                else:
                    st.divider()
                    st.caption("⚠️ Cannot delete the last chat")
    
    st.divider()
    
    # Language Toggle
    st.subheader("🌐 Response Language")
    
    col_lang1, col_lang2 = st.columns(2)
    with col_lang1:
        if st.button(
            "🇺🇸 English", 
            key="lang_en",
            use_container_width=True,
            type="primary" if st.session_state.response_language == "English" else "secondary"
        ):
            st.session_state.response_language = "English"
            st.rerun()
    
    with col_lang2:
        if st.button(
            "🇮🇳 हिंदी", 
            key="lang_hi",
            use_container_width=True,
            type="primary" if st.session_state.response_language == "Hindi" else "secondary"
        ):
            st.session_state.response_language = "Hindi"
            st.rerun()
    
    st.caption(f"Currently responding in: **{st.session_state.response_language}**")
    
    st.divider()
    
    # System status
    st.subheader("⚙️ System Status")
    if ENHANCED_PROCESSING:
        st.success("✅ Enhanced Processing")
    else:
        st.warning("⚠️ Basic Mode")
    
    if MULTIMODAL_AVAILABLE:
        st.success("✅ Multimodal Vision")
    
    # Statistics for active chat
    if active_chat["chat_history"]:
        st.metric("Messages in Chat", len(active_chat["chat_history"]))
    
    st.divider()
    
    # Help section
    with st.expander("ℹ️ Tips for Better Results"):
        st.markdown("""
        - **Be specific**: Mention disease names, symptoms
        - **Use context**: Refer to previous messages
        - **Ask about**: Symptoms, treatments, prevention
        
        **Examples**: 
        - "What are symptoms of late blight?"
        - "How to prevent blackleg?"
        - "Treatment for ring rot?"
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

@st.cache_resource
def load_multimodal():
    if MULTIMODAL_AVAILABLE:
        # Assumes the index is in the default location
        return create_multimodal_generator("faiss_index_multimodal")
    return None

qa_chain = load_chain()
query_processor, context_filter = load_processors()
multimodal_generator = load_multimodal()

# --- Display existing messages from active chat ---
for role, content in active_chat["messages"]:
    with st.chat_message(role):
        st.markdown(content)

# --- Handle User Input ---
user_query = st.chat_input("Ask about potato diseases...")

if user_query:
    # Auto-rename chat if it's the first message
    if len(active_chat["messages"]) == 0:
        auto_rename_chat(st.session_state.active_chat_id, user_query)
    
    # Add user message to active chat
    active_chat["messages"].append(("user", user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    # Unified RAG processing
    with st.spinner("Thinking..."):
        try:
            # Enhanced query processing if available
            if ENHANCED_PROCESSING and query_processor:
                processed_query = query_processor.preprocess_question(
                    user_query, 
                    active_chat["chat_history"]
                )
                query_to_use = processed_query['primary_query']
            else:
                query_to_use = user_query
            
            # Add language instruction to the query
            # We add this to the query sent to LLM, but use raw query for retrieval if needed
            language_instruction = ""
            if st.session_state.response_language == "Hindi":
                language_instruction = "\n\nIMPORTANT: Respond to this question in Hindi (हिंदी में उत्तर दें)."
                query_with_language = user_query + language_instruction
            else:
                query_with_language = user_query
            
            # 1. Retrieve Contexts first (Needed for Images)
            retrieval_result = qa_chain.invoke({
                "question": query_with_language,
                "chat_history": active_chat["chat_history"]
            })
            
            source_documents = retrieval_result.get("source_documents", [])
            
            # Enhanced context filtering
            if ENHANCED_PROCESSING and context_filter and source_documents:
                filtered_docs = context_filter.filter_contexts(
                    source_documents, 
                    user_query, 
                    max_contexts=6
                )
                source_documents = filtered_docs

            # 2. Stream the AI response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                
                # Stream response chunks
                for chunk in qa_chain.stream({
                    "question": query_with_language,
                    "chat_history": active_chat["chat_history"]
                }):
                    if 'answer' in chunk:
                        full_response += chunk['answer']
                        response_placeholder.markdown(full_response + "▌")
                
                # Final display without cursor
                response_placeholder.markdown(full_response)
                ai_response = full_response
                
                # 3. Multimodal Image Logic (After streaming text)
                images_to_display = []
                if multimodal_generator:
                    multimodal_result = multimodal_generator.generate_multimodal_response(
                        question=user_query,
                        text_answer=ai_response,
                        retrieved_contexts=source_documents
                    )
                    
                    if multimodal_result['has_images']:
                        images_to_display = multimodal_result['images']
                        
                        st.markdown("### 📸 Visual References" if st.session_state.response_language == "English" else "### 📸 दृश्य संदर्भ (Visual References)")
                        
                        cols = st.columns(min(len(images_to_display), 3))
                        for idx, img in enumerate(images_to_display):
                            col_idx = idx % 3
                            with cols[col_idx]:
                                if os.path.exists(img['path']):
                                    st.image(
                                        img['path'], 
                                        caption=f"{img.get('image_name', 'Image')}",
                                        use_container_width=True
                                    )
                                    with st.expander("Details"):
                                        st.caption(img['caption'])
                                else:
                                    st.error(f"Image not found: {img['path']}")

                # 4. Display Sources
                if source_documents:
                    st.markdown("---")
                    with st.expander(f"📚 View Sources ({len(source_documents)} found)"):
                        for i, doc in enumerate(source_documents):
                            source_name = doc.metadata.get('source', 'Unknown Source')
                            doc_type = doc.metadata.get('type', 'text')
                            icon = "🖼️" if doc_type == 'image_description' else "📄"

                            st.markdown(f"**{icon} Source {i+1}:** `{source_name}`")
                            
                            content_preview = doc.page_content[:300].replace('\n', ' ')
                            if len(doc.page_content) > 300:
                                content_preview += "..."
                            st.markdown(f"> {content_preview}")
                            
                            if i < len(source_documents) - 1:
                                st.divider()
                else:
                    st.info("ℹ️ No specific sources found. Response based on general knowledge.")

            # Add AI response to active chat
            active_chat["messages"].append(("assistant", ai_response))

            # Add this Q&A to the active chat's memory
            active_chat["chat_history"].append((user_query, ai_response))
            
            # Save to disk after each message
            save_chats_to_disk()
            
        except Exception as e:
            st.error(f"Sorry, I encountered an error: {str(e)}")
            st.info("Please try rephrasing your question or contact support if the issue persists.")
            # import traceback
            # st.text(traceback.format_exc())