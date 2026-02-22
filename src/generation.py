import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import BaseMessage, HumanMessage, AIMessage, Document
from langchain_core.prompts import PromptTemplate
from typing import List, Tuple, Dict
import time
from src.logging_utils import setup_logger, timer, log_timing, log_generation_metrics

load_dotenv()

# Initialize logger
logger = setup_logger('generation')

class ImprovedConversationalChain:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, streaming=True)
        
        # Use ConversationSummaryBufferMemory for better memory management
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            memory_key='chat_history',
            return_messages=True,
            output_key='answer',
            max_token_limit=1000  # Prevent memory overflow
        )
        
        # Custom prompt for better context handling
        self.qa_prompt = PromptTemplate(
            template="""You are Aloo Sahayak, an expert agricultural AI assistant specializing in potato diseases, nutrition, and cultivation practices.

INSTRUCTIONS:
1. If the user sends a simple greeting (hi, hello, hey, thanks, bye), respond politely and briefly.
2. For questions about potatoes (diseases, nutrition, fertilizers, cultivation), use the provided context to give accurate, detailed answers.
3. If the context doesn't contain sufficient information to answer the question, say "I don't have enough information about this in my knowledge base."
4. NEVER return generic greetings as answers to factual questions - always attempt to use the context first.
5. Be professional, helpful, and cite sources when available.

Conversation History:
{chat_history}

Context from Knowledge Base:
{context}

User Question: {question}

Provide a comprehensive answer based on the context. For factual questions, always prioritize using the retrieved context over general knowledge.

Answer:""",
            input_variables=["context", "question", "chat_history"]
        )
        
        # Create the chain with improved settings
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            return_generated_question=True,
            combine_docs_chain_kwargs={"prompt": self.qa_prompt},
            verbose=False,
            max_tokens_limit=4000  # Control response length
        )
    
    def invoke(self, inputs):
        """Enhanced invoke method"""
        invoke_start = time.perf_counter()
        
        try:
            # Extract question and chat_history from inputs
            question = inputs.get("question", "")
            external_chat_history = inputs.get("chat_history", [])
            
            # Sync memory with external history if provided
            if external_chat_history:
                sync_start = time.perf_counter()
                self._sync_memory_with_external_history(external_chat_history)
                sync_elapsed = time.perf_counter() - sync_start
                log_timing(logger, "MEMORY_SYNC", {
                    'duration_ms': round(sync_elapsed * 1000, 2),
                    'history_items': len(external_chat_history)
                })
            
            # Invoke the chain to get text answer
            chain_start = time.perf_counter()
            result = self.chain.invoke({"question": question})
            chain_elapsed = time.perf_counter() - chain_start
            
            log_timing(logger, "CHAIN_INVOCATION", {
                'duration_ms': round(chain_elapsed * 1000, 2),
                'answer_length': len(result.get('answer', '')),
                'source_docs': len(result.get('source_documents', []))
            })
            
            total_elapsed = time.perf_counter() - invoke_start
            log_generation_metrics(logger, None, total_elapsed)
            
            return result
            
        except Exception as e:
            elapsed = time.perf_counter() - invoke_start
            logger.error(f"Error in chain invocation: {e}")
            log_timing(logger, "INVOKE_ERROR", {
                'duration_ms': round(elapsed * 1000, 2),
                'error': type(e).__name__
            })
            return {
                "answer": "I apologize, but I encountered an error processing your question. Please try again.",
                "source_documents": [],
                "generated_question": question
            }
    
    def stream(self, inputs):
        """Stream the response for real-time display and return source documents"""
        stream_start = time.perf_counter()
        chunk_count = 0
        
        try:
            # Extract question and chat_history from inputs
            question = inputs.get("question", "")
            external_chat_history = inputs.get("chat_history", [])
            
            # Add external chat history to memory if provided
            if external_chat_history:
                sync_start = time.perf_counter()
                self._sync_memory_with_external_history(external_chat_history)
                sync_elapsed = time.perf_counter() - sync_start
                log_timing(logger, "MEMORY_SYNC_STREAM", {
                    'duration_ms': round(sync_elapsed * 1000, 2),
                    'history_items': len(external_chat_history)
                })
            
            # Collect full answer and source documents
            full_answer = ""
            source_docs = []
            chunk_start = time.perf_counter()
            
            # Stream the response
            for chunk in self.chain.stream({"question": question}):
                chunk_count += 1
                
                if 'answer' in chunk:
                    text_chunk = chunk['answer']
                    full_answer += text_chunk
                    yield {'type': 'chunk', 'content': text_chunk}
                elif isinstance(chunk, str):
                    full_answer += chunk
                    yield {'type': 'chunk', 'content': chunk}
                
                # Capture source documents if available
                if 'source_documents' in chunk:
                    source_docs = chunk['source_documents']
            
            chunk_elapsed = time.perf_counter() - chunk_start
            log_timing(logger, "STREAMING_CHUNKS", {
                'duration_ms': round(chunk_elapsed * 1000, 2),
                'chunk_count': chunk_count,
                'answer_length': len(full_answer)
            })
            
            total_stream_elapsed = time.perf_counter() - stream_start
            log_generation_metrics(logger, None, total_stream_elapsed, 
                                  chunk_count=chunk_count, token_count=len(full_answer.split()))
            
            # After streaming completes, yield the source documents
            yield {
                'type': 'complete',
                'answer': full_answer,
                'source_documents': source_docs
            }
            
        except Exception as e:
            elapsed = time.perf_counter() - stream_start
            logger.error(f"Error in chain streaming: {e}")
            log_timing(logger, "STREAM_ERROR", {
                'duration_ms': round(elapsed * 1000, 2),
                'chunks_sent': chunk_count,
                'error': type(e).__name__
            })
            yield {
                'type': 'error',
                'content': "I apologize, but I encountered an error processing your question. Please try again."
            }
    
    def _sync_memory_with_external_history(self, external_history: List[Tuple[str, str]]):
        """Sync external chat history with internal memory"""
        # Clear current memory to avoid duplication
        self.memory.clear()
        
        # Add each conversation turn to memory
        for human_msg, ai_msg in external_history[-5:]:  # Keep last 5 exchanges
            self.memory.chat_memory.add_user_message(human_msg)
            self.memory.chat_memory.add_ai_message(ai_msg)

def create_conversational_chain(retriever):
    """
    Creates an improved conversational RAG chain with better memory management.
    """
    print("Initializing improved conversational RAG chain...")
    
    chain = ImprovedConversationalChain(retriever)
    
    print("Improved conversational RAG chain created successfully.")
    return chain