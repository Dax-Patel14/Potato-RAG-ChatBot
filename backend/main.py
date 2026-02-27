import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
from typing import List
import asyncio
import threading
import time
import logging
import json

from backend.config import CORS_ORIGINS, API_HOST, API_PORT, DATABASE_TYPE, DEBUG
from backend.schemas import (
    ChatCreate, ChatRename, ChatResponse, ChatDetailResponse,
    MessageCreate, MessageResponse, StreamMessage, HealthResponse
)
from src.chat_db import (
    init_db, add_chat, get_chats, get_messages, 
    add_message, rename_chat as db_rename_chat, 
    delete_chat as db_delete_chat, get_chat_by_id
)
from src.generation import create_conversational_chain
from src.retrieval import load_retriever_from_disk
from src.logging_utils import setup_logger, log_timing, log_query_start, log_query_complete

# Initialize logger
api_logger = setup_logger('api')

# Initialize FastAPI app
app = FastAPI(
    title="Aloo Sahayak API",
    description="Backend API for Potato Disease Assistant with RAG",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if isinstance(CORS_ORIGINS, list) else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG chain instance
qa_chain = None
retriever = None

# Startup and Shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize database and load RAG chain on startup"""
    global qa_chain, retriever
    
    print(" Starting Aloo Sahayak API...")
    
    # Initialize database
    try:
        init_db()
        print(" Database initialized successfully")
    except Exception as e:
        print(f" Database initialization failed: {e}")
        raise
    
    # Load retriever and RAG chain
    try:
        retriever = load_retriever_from_disk()
        qa_chain = create_conversational_chain(retriever)
        print(" RAG chain loaded successfully")
    except Exception as e:
        print(f" RAG chain loading failed: {e}")
        raise
    
    print(" API startup complete!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print(" Shutting down Aloo Sahayak API...")

# ==================== Health Check ====================
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check API health status"""
    return HealthResponse(
        status="healthy",
        message="API is running",
        database=DATABASE_TYPE
    )

# ==================== Chat Endpoints ====================

@app.get("/api/chats", response_model=List[ChatResponse])
async def list_chats():
    """Get all chats"""
    try:
        chats_data = get_chats()
        chats = []
        
        for chat_id, name, created_at in chats_data:
            messages = get_messages(chat_id)
            chats.append(ChatResponse(
                id=chat_id,
                name=name,
                created_at=created_at,
                message_count=len(messages) // 2  # Divide by 2 (user + assistant)
            ))
        
        return chats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chats", response_model=ChatResponse)
async def create_chat(chat: ChatCreate):
    """Create a new chat"""
    try:
        chat_id = str(uuid.uuid4())
        add_chat(chat_id, chat.name)
        
        return ChatResponse(
            id=chat_id,
            name=chat.name,
            created_at=get_chats()[0][2],  # Get created_at from newly created chat
            message_count=0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chats/{chat_id}", response_model=ChatDetailResponse)
async def get_chat(chat_id: str):
    """Get chat details with messages"""
    try:
        chats = get_chats()
        chat_found = False
        chat_name = None
        created_at = None
        
        for cid, name, created in chats:
            if cid == chat_id:
                chat_found = True
                chat_name = name
                created_at = created
                break
        
        if not chat_found:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        messages = get_messages(chat_id)
        message_list = []
        for i, msg in enumerate(messages):
            sender = msg[0]
            content = msg[1]
            timestamp = msg[2]
            metadata = msg[3] if len(msg) > 3 else None
            message_list.append(
                MessageResponse(
                    id=i,
                    sender=sender,
                    content=content,
                    timestamp=timestamp,
                    metadata=metadata
                )
            )
        
        return ChatDetailResponse(
            id=chat_id,
            name=chat_name,
            created_at=created_at,
            messages=message_list
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/chats/{chat_id}", response_model=ChatResponse)
async def update_chat(chat_id: str, chat: ChatRename):
    """Rename a chat"""
    try:
        chats = get_chats()
        chat_found = False
        
        for cid, name, created in chats:
            if cid == chat_id:
                chat_found = True
                created_at = created
                break
        
        if not chat_found:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        db_rename_chat(chat_id, chat.new_name)
        
        return ChatResponse(
            id=chat_id,
            name=chat.new_name,
            created_at=created_at
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a chat"""
    try:
        db_delete_chat(chat_id)
        return {"status": "success", "message": "Chat deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Message Endpoints ====================

@app.get("/api/chats/{chat_id}/messages", response_model=List[MessageResponse])
async def get_chat_messages(chat_id: str):
    """Get all messages for a chat"""
    try:
        messages = get_messages(chat_id)
        
        result = []
        for i, msg in enumerate(messages):
            sender = msg[0]
            content = msg[1]
            timestamp = msg[2]
            metadata = msg[3] if len(msg) > 3 else None
            result.append(MessageResponse(id=i, sender=sender, content=content, timestamp=timestamp, metadata=metadata))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chats/{chat_id}/messages")
async def send_message(chat_id: str, message: MessageCreate):
    """Send message and get AI response (non-streaming)"""
    request_id = str(uuid.uuid4())[:8]
    request_start = time.perf_counter()
    
    try:
        log_query_start(api_logger, request_id, message.content[:100])
        
        if qa_chain is None:
            raise HTTPException(status_code=500, detail="RAG chain not loaded")
        
        user_query = message.content
        
        # Add user message to database
        db_start = time.perf_counter()
        add_message(chat_id, "user", user_query)
        db_elapsed = time.perf_counter() - db_start
        
        log_timing(api_logger, "DB_ADD_USER_MESSAGE", {
            'duration_ms': round(db_elapsed * 1000, 2)
        })
        
        # Get chat history for context
        hist_start = time.perf_counter()
        messages = get_messages(chat_id)
        chat_history = [
            (msg[0], msg[1]) 
            for msg in messages[:-1]  # Exclude latest user message
            if msg[0] in ["user", "assistant"]
        ]
        
        # Reconstruct alternating pairs for context
        reconstructed_history = []
        for i in range(0, len(chat_history), 2):
            if i + 1 < len(chat_history):
                reconstructed_history.append((chat_history[i][1], chat_history[i+1][1]))
        
        hist_elapsed = time.perf_counter() - hist_start
        log_timing(api_logger, "HISTORY_RETRIEVAL", {
            'duration_ms': round(hist_elapsed * 1000, 2),
            'history_items': len(reconstructed_history)
        })
        
        # Add language instruction
        if message.language == "Hindi":
            query = user_query + "\n\nIMPORTANT: Respond to this question in Hindi (हिंदी में उत्तर दें)."
        else:
            query = user_query
        
        # Get AI response (measure time)
        chain_start = time.perf_counter()
        result = qa_chain.invoke({
            "question": query,
            "chat_history": reconstructed_history
        })
        chain_elapsed = time.perf_counter() - chain_start
        
        log_timing(api_logger, "AI_CHAIN_INVOCATION", {
            'duration_ms': round(chain_elapsed * 1000, 2)
        })
        
        ai_response = result.get("answer", "")
        source_docs = result.get("source_documents", [])
        
        # Serialize source documents to JSON-friendly structure (force JSON-safe types)
        serialized_sources = []
        for doc in source_docs:
            try:
                raw_meta = doc.metadata if hasattr(doc, 'metadata') else {}
                src = raw_meta.get('source', None)
                page_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                if src is not None:
                    src = str(src)
                safe_meta = {}
                for k, v in (raw_meta or {}).items():
                    try:
                        json.dumps(v)
                        safe_meta[str(k)] = v
                    except (TypeError, ValueError):
                        safe_meta[str(k)] = str(v)
            except Exception:
                src = None
                safe_meta = {}
                page_content = str(doc)

            serialized_sources.append({
                "source": src,
                "page_content": page_content,
                "metadata": safe_meta
            })

        # Add AI message to database
        db_ai_start = time.perf_counter()
        add_message(chat_id, "assistant", ai_response, metadata={"source_documents": serialized_sources})
        db_ai_elapsed = time.perf_counter() - db_ai_start
        
        log_timing(api_logger, "DB_ADD_AI_MESSAGE", {
            'duration_ms': round(db_ai_elapsed * 1000, 2),
            'response_length': len(ai_response)
        })

        total_time = time.perf_counter() - request_start
        log_query_complete(api_logger, request_id, total_time, "SUCCESS")

        return {
            "user_message": user_query,
            "ai_response": ai_response,
            "sources_count": len(serialized_sources),
            "source_documents": serialized_sources,
            "language": message.language,
            "timings": {"total_ms": round(total_time * 1000, 2)}
        }
    except HTTPException:
        raise
    except Exception as e:
        total_time = time.perf_counter() - request_start
        log_query_complete(api_logger, request_id, total_time, f"FAILED: {type(e).__name__}")
        api_logger.error(f"Error in send_message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== WebSocket for Streaming ====================

@app.websocket("/api/ws/chats/{chat_id}/stream")
async def websocket_stream(websocket: WebSocket, chat_id: str):
    """WebSocket endpoint for real-time message streaming"""
    request_id = str(uuid.uuid4())[:8]
    request_start = time.perf_counter()
    
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            user_query = data.get("message", "")
            language = data.get("language", "English")
            
            if not user_query:
                await websocket.send_json({
                    "type": "error",
                    "error": "Message content is required"
                })
                continue
            
            log_query_start(api_logger, request_id, user_query[:100])
            
            # Add user message to database
            db_start = time.perf_counter()
            add_message(chat_id, "user", user_query)
            db_elapsed = time.perf_counter() - db_start
            
            log_timing(api_logger, "DB_ADD_USER_WS", {
                'duration_ms': round(db_elapsed * 1000, 2)
            })
            
            # Notify client that user message received
            await websocket.send_json({
                "type": "user_received",
                "message": user_query
            })
            
            # Get chat history
            hist_start = time.perf_counter()
            messages = get_messages(chat_id)
            chat_history = [
                (msg[0], msg[1])
                for msg in messages[:-1]  # Exclude latest user message
                if msg[0] in ["user", "assistant"]
            ]
            
            # Reconstruct alternating pairs
            reconstructed_history = []
            for i in range(0, len(chat_history), 2):
                if i + 1 < len(chat_history):
                    reconstructed_history.append((chat_history[i][1], chat_history[i+1][1]))
            
            hist_elapsed = time.perf_counter() - hist_start
            log_timing(api_logger, "HISTORY_RETRIEVAL_WS", {
                'duration_ms': round(hist_elapsed * 1000, 2),
                'history_items': len(reconstructed_history)
            })
            
            # Add language instruction
            if language == "Hindi":
                query = user_query + "\n\nIMPORTANT: Respond to this question in Hindi (हिंदी में उत्तर दें)."
            else:
                query = user_query
            
            # Stream the response and measure timings
            stream_start = time.perf_counter()
            full_response = ""
            source_docs = []
            first_chunk_time = None
            chunk_count = 0
            # Initialize here so it is always defined even if the stream errors out
            # before the 'complete' message arrives — previously caused UnboundLocalError
            # which silently dropped the AI message from the database on every error.
            serialized_sources = []

            # Use an asyncio.Queue to receive chunks from a background thread.
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def run_blocking_stream():
                try:
                    for stream_output in qa_chain.stream({
                        "question": query,
                        "chat_history": reconstructed_history
                    }):
                        # Push each chunk into the asyncio queue from this thread
                        loop.call_soon_threadsafe(queue.put_nowait, stream_output)
                except Exception as e:
                    loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "content": str(e)})

            # Start background thread for blocking streaming work
            thread = threading.Thread(target=run_blocking_stream, daemon=True)
            thread.start()

            try:
                while True:
                    # Wait for the next chunk from the background thread
                    stream_output = await queue.get()

                    if stream_output.get('type') == 'chunk':
                        chunk_content = stream_output['content']
                        chunk_count += 1
                        if first_chunk_time is None:
                            first_chunk_time = time.perf_counter() - stream_start
                        full_response += chunk_content
                        await websocket.send_json({
                            "type": "chunk",
                            "content": chunk_content
                        })

                    elif stream_output.get('type') == 'complete':
                        full_response = stream_output.get('answer', full_response)
                        source_docs = stream_output.get('source_documents', [])
                        total_stream_elapsed = time.perf_counter() - stream_start

                        # Serialize source documents to JSON-friendly structure.
                        # Force every metadata value to a JSON-safe type so that
                        # json.dumps() in add_message() never silently drops sources
                        # (e.g. PurePosixPath objects are not serializable by default).
                        serialized_sources = []
                        for doc in source_docs:
                            try:
                                raw_meta = doc.metadata if hasattr(doc, 'metadata') else {}
                                src = raw_meta.get('source', None)
                                page_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                                # Convert source path to plain string
                                if src is not None:
                                    src = str(src)
                                # Sanitize every metadata value to be JSON-serializable
                                safe_meta = {}
                                for k, v in (raw_meta or {}).items():
                                    try:
                                        json.dumps(v)
                                        safe_meta[str(k)] = v
                                    except (TypeError, ValueError):
                                        safe_meta[str(k)] = str(v)
                            except Exception:
                                src = None
                                safe_meta = {}
                                page_content = str(doc)

                            serialized_sources.append({
                                "source": src,
                                "page_content": page_content,
                                "metadata": safe_meta
                            })

                        log_timing(api_logger, "STREAMING_COMPLETE", {
                            'duration_ms': round(total_stream_elapsed * 1000, 2),
                            'chunks': chunk_count,
                            'response_length': len(full_response),
                            'first_chunk_ms': round(first_chunk_time * 1000, 2) if first_chunk_time else None
                        })

                        await websocket.send_json({
                            "type": "complete",
                            "content": full_response,
                            "sources_count": len(serialized_sources),
                            "source_documents": serialized_sources,
                            "timings": {
                                "first_chunk_ms": round(first_chunk_time * 1000, 2) if first_chunk_time else None,
                                "total_ms": round(total_stream_elapsed * 1000, 2)
                            }
                        })
                        break

                    elif stream_output.get('type') == 'error':
                        await websocket.send_json({
                            "type": "error",
                            "error": stream_output.get('content')
                        })
                        break

                # Add AI response to database after streaming completes (include sources metadata)
                db_ai_start = time.perf_counter()
                add_message(chat_id, "assistant", full_response, metadata={"source_documents": serialized_sources})
                db_ai_elapsed = time.perf_counter() - db_ai_start
                
                log_timing(api_logger, "DB_ADD_AI_WS", {
                    'duration_ms': round(db_ai_elapsed * 1000, 2),
                    'response_length': len(full_response)
                })
                
                total_elapsed = time.perf_counter() - request_start
                log_query_complete(api_logger, request_id, total_elapsed, "SUCCESS")

            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "error": f"Error generating response: {str(e)}"
                })
                total_elapsed = time.perf_counter() - request_start
                log_query_complete(api_logger, request_id, total_elapsed, f"FAILED: {type(e).__name__}")
    
    except WebSocketDisconnect:
        api_logger.info(f"WebSocket client disconnected - request_id={request_id}")
    except Exception as e:
        api_logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass

# ==================== Error Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "detail": exc.detail
        },
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "status_code": 500,
            "detail": "Internal server error"
        },
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Aloo Sahayak API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "status": "running"
    }

if __name__ == "__main__":
    import uvicorn
    
    print(f"Starting API on {API_HOST}:{API_PORT}")
    print(f"Database: {DATABASE_TYPE}")
    print(f"Debug: {DEBUG}")
    print(f"API Documentation: http://{API_HOST}:{API_PORT}/api/docs")
    
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        debug=DEBUG,
        reload=DEBUG
    )