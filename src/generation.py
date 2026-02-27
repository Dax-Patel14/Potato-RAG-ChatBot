import os
import queue
import threading
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import BaseMessage, HumanMessage, AIMessage, Document, LLMResult
from langchain_core.prompts import PromptTemplate
from typing import Any, List, Tuple, Dict
import time
from src.logging_utils import setup_logger, timer, log_timing, log_generation_metrics

load_dotenv()

# Initialize logger
logger = setup_logger('generation')


class TokenStreamingHandler(BaseCallbackHandler):
    """
    Callback handler that pushes each generated token into a queue.Queue.

    on_llm_new_token() fires ONLY for LLM instances that have streaming=True.
    By giving the condense/question-rephrasing LLM streaming=False, its tokens
    are never sent here — only the final answer tokens flow through.
    """

    def __init__(self, token_queue: queue.Queue):
        super().__init__()
        self.token_queue = token_queue

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Called for every new token from a streaming=True LLM."""
        self.token_queue.put({"type": "token", "content": token})

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        self.token_queue.put({"type": "error", "content": str(error)})


class CondenserTimingHandler(BaseCallbackHandler):
    """
    Attaches to condense_llm (streaming=False) to log the duration of the
    question-condensation step. Previously this step was completely invisible
    in logs, causing confusing 44-86s gaps between MEMORY_SYNC and
    ENSEMBLE_RETRIEVAL with no explanation.
    """

    def __init__(self):
        super().__init__()
        self._start: float = 0.0

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        self._start = time.perf_counter()
        logger.info("TIMING | CONDENSE_QUESTION_LLM          | Started (rephrasing follow-up as standalone question)")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        elapsed = time.perf_counter() - self._start
        log_timing(logger, "CONDENSE_QUESTION_LLM", {
            'duration_ms': round(elapsed * 1000, 2)
        })

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        elapsed = time.perf_counter() - self._start
        log_timing(logger, "CONDENSE_QUESTION_LLM_ERROR", {
            'duration_ms': round(elapsed * 1000, 2),
            'error': str(error)[:80]
        })


class SourceDocumentCaptureHandler(BaseCallbackHandler):
    """
    Captures source documents directly from the retriever via on_retriever_end().

    This is more reliable than reading from result_container['result']['source_documents']
    because in some LangChain versions, ConversationalRetrievalChain.invoke() does not
    consistently surface source_documents at the top level of the output dict when
    memory is attached to the chain (output_key='answer' restricts what is returned).

    on_retriever_end() fires at the exact moment the retriever returns documents,
    completely independent of how invoke() structures its final output.
    """

    def __init__(self):
        super().__init__()
        self.source_docs: List[Any] = []

    def on_retriever_end(self, documents: List[Any], **kwargs: Any) -> None:
        """Called when the retriever finishes. Captures the retrieved documents."""
        self.source_docs = documents
        logger.info(f"TIMING | SOURCE_DOCS_CAPTURED            | count={len(documents)}")


class ImprovedConversationalChain:
    def __init__(self, retriever):
        self.retriever = retriever

        # Answer-generation LLM: streaming=True so on_llm_new_token fires per token.
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, streaming=True)

        # Condense / infrastructure LLM: streaming=False so its tokens are NEVER sent
        # to the streaming callback handler — prevents sending the rephrased question
        # text to the user before the actual answer.
        # CondenserTimingHandler makes the silent condense step visible in logs.
        self.condense_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            streaming=False,
            callbacks=[CondenserTimingHandler()]
        )

        # Memory uses the non-streaming LLM so summarization calls don't leak tokens.
        self.memory = ConversationSummaryBufferMemory(
            llm=self.condense_llm,
            memory_key='chat_history',
            return_messages=True,
            output_key='answer',
            max_token_limit=1000
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
        
        # condense_question_llm handles the "rephrase follow-up as standalone question"
        # step using the non-streaming LLM, so those tokens never reach the handler.
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            condense_question_llm=self.condense_llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            return_generated_question=True,
            combine_docs_chain_kwargs={"prompt": self.qa_prompt},
            verbose=False,
            max_tokens_limit=4000
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
        """
        True token-by-token streaming via TokenStreamingHandler.

        How it works:
        1. A TokenStreamingHandler is created with a queue.Queue.
        2. self.chain.invoke() runs in a background thread, passing the handler
           via LangChain's config={"callbacks": [handler]}.
        3. Because self.llm has streaming=True, on_llm_new_token() fires for each
           answer token and puts it into the queue immediately.
        4. Because self.condense_llm has streaming=False, the question-condensation
           step never fires on_llm_new_token — those tokens stay silent.
        5. This generator yields each token from the queue as it arrives, giving the
           WebSocket handler real tokens to send to the browser in real-time.

        Previously self.chain.stream() was used, which buffers the entire answer
        inside the chain and yields a single dict at the end — that is why
        chunk_count=1 appeared in the logs with first_chunk_ms == total_ms.
        """
        stream_start = time.perf_counter()
        chunk_count = 0

        question = inputs.get("question", "")
        external_chat_history = inputs.get("chat_history", [])

        if external_chat_history:
            sync_start = time.perf_counter()
            self._sync_memory_with_external_history(external_chat_history)
            sync_elapsed = time.perf_counter() - sync_start
            log_timing(logger, "MEMORY_SYNC_STREAM", {
                'duration_ms': round(sync_elapsed * 1000, 2),
                'history_items': len(external_chat_history)
            })
        else:
            # No external history means this is a fresh conversation.
            # Explicitly clear chain memory so stale state from a previous
            # conversation doesn't cause an unnecessary condense_llm API call.
            # Bug observed: query 1f3d876f had history_items=0 but still triggered
            # a ~33s condense call because memory held data from the prior session.
            self.memory.clear()
            logger.info("TIMING | MEMORY_CLEAR                   | Cleared stale chain memory for new conversation")

        # Per-request token queue — thread-safe bridge between the chain thread
        # and this generator.
        token_queue: queue.Queue = queue.Queue()
        handler = TokenStreamingHandler(token_queue)
        # Captures source documents directly from the retriever callback — more
        # reliable than reading result_container['result']['source_documents'] since
        # ConversationalRetrievalChain.invoke() with memory attached sometimes strips
        # source_documents from the top-level output dict (output_key='answer' effect).
        source_capture = SourceDocumentCaptureHandler()
        result_container: Dict[str, Any] = {}

        def run_chain():
            """Runs in a background thread. Puts chain_done sentinel when finished."""
            try:
                result = self.chain.invoke(
                    {"question": question},
                    config={"callbacks": [handler, source_capture]}
                )
                result_container['result'] = result
            except Exception as exc:
                result_container['error'] = str(exc)
                token_queue.put({"type": "error", "content": str(exc)})
            finally:
                # Always signal completion so the generator is never stuck
                token_queue.put({"type": "chain_done"})

        chain_thread = threading.Thread(target=run_chain, daemon=True)
        chain_thread.start()

        full_answer = ""
        chunk_start = time.perf_counter()

        try:
            while True:
                # Block until the next item arrives (5-minute safety timeout)
                item = token_queue.get(timeout=300)

                if item["type"] == "token":
                    chunk_count += 1
                    full_answer += item["content"]
                    yield {"type": "chunk", "content": item["content"]}

                elif item["type"] == "chain_done":
                    break

                elif item["type"] == "error":
                    logger.error(f"Token stream error from chain thread: {item['content']}")
                    yield {"type": "error", "content": "I apologize, but I encountered an error processing your question. Please try again."}
                    chain_thread.join(timeout=5)
                    return

        except queue.Empty:
            logger.error("Token queue timed out after 300s — chain thread may have hung")
            yield {"type": "error", "content": "Request timed out. Please try again."}
            return

        chain_thread.join(timeout=10)

        chunk_elapsed = time.perf_counter() - chunk_start
        log_timing(logger, "STREAMING_CHUNKS", {
            'duration_ms': round(chunk_elapsed * 1000, 2),
            'chunk_count': chunk_count,
            'answer_length': len(full_answer)
        })

        total_stream_elapsed = time.perf_counter() - stream_start
        log_generation_metrics(logger, None, total_stream_elapsed,
                               chunk_count=chunk_count, token_count=len(full_answer.split()))

        # source_capture.source_docs is populated via on_retriever_end() callback,
        # which fires when the retriever returns docs — independent of invoke() output.
        # Fall back to result_container if for any reason capture is empty.
        result = result_container.get('result', {})
        source_docs = source_capture.source_docs or result.get('source_documents', [])

        log_timing(logger, "SOURCE_DOCS_FINAL", {
            'captured_count': len(source_capture.source_docs),
            'result_count': len(result.get('source_documents', [])),
            'final_count': len(source_docs)
        })

        yield {
            'type': 'complete',
            'answer': full_answer,
            'source_documents': source_docs
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