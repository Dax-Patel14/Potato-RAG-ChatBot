from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever # To combine semantic + Keyword based
from langchain_community.retrievers import BM25Retriever # Keyword based
from langchain.schema import Document, BaseRetriever
from typing import List, Any, Optional
from pydantic import Field, PrivateAttr

# Point this to the new multimodal index directory
FAISS_INDEX_PATH = "faiss_index_multimodal"

class EnhancedRetriever(BaseRetriever):
    """Enhanced retriever that inherits from BaseRetriever for LangChain compatibility"""
    
    # Pydantic configuration to allow complex objects like FAISS and ChatOpenAI
    model_config = {"arbitrary_types_allowed": True}
    
    # Public fields (can be passed in init)
    faiss_index_path: str = FAISS_INDEX_PATH
    
    # Private attributes (internal state, not exposed to Pydantic validation)
    _embeddings: Any = PrivateAttr()
    _llm: Any = PrivateAttr()
    _vector_store: Any = PrivateAttr()
    _semantic_retriever: Any = PrivateAttr()
    _bm25_retriever: Optional[Any] = PrivateAttr(default=None)
    _ensemble_retriever: Any = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize components using the path defined in self.faiss_index_path
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self._llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
        # Load FAISS vector store
        self._vector_store = FAISS.load_local(
            self.faiss_index_path, 
            self._embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Create semantic retriever with more candidates
        self._semantic_retriever = self._vector_store.as_retriever(
            search_kwargs={"k": 12}
        )
        
        # Try to create BM25 retriever for hybrid search
        try:
            self._setup_bm25_retriever()
            if self._bm25_retriever:
                self._ensemble_retriever = EnsembleRetriever(
                    retrievers=[self._semantic_retriever, self._bm25_retriever],
                    weights=[0.7, 0.3]
                )
            else:
                self._ensemble_retriever = self._semantic_retriever
        except Exception:
            self._ensemble_retriever = self._semantic_retriever
            self._bm25_retriever = None

    # method to define BM25 retriever
    def _setup_bm25_retriever(self):
        """Setup BM25 retriever for keyword search"""
        try:
            # Get all documents from FAISS
            all_docs = []
            docstore = self._vector_store.docstore
            
            for doc_id in self._vector_store.index_to_docstore_id.values():
                doc = docstore.search(doc_id)
                all_docs.append(doc)
            
            # Create BM25 retriever
            self._bm25_retriever = BM25Retriever.from_documents(
                all_docs,
                k=6
            )
            
        except Exception as e:
            print(f"BM25 setup failed, using semantic only: {e}")
            self._bm25_retriever = None
    
    def preprocess_query(self, question: str) -> str:
        """Enhance query with domain knowledge"""
        question_lower = question.lower()
        
        # Disease name expansions
        expansions = {
            'late blight': 'late blight phytophthora infestans',
            'early blight': 'early blight alternaria solani',
            'blackleg': 'blackleg soft rot pectobacterium bacterial',
            'ring rot': 'ring rot bacterial clavibacter',
            'scab': 'scab streptomyces bacterial',
            'dry rot': 'dry rot fusarium fungal',
            'soft rot': 'soft rot bacterial pectobacterium',
            'wilt': 'wilt verticillium bacterial fungal'
        }
        
        enhanced_question = question
        for disease, expansion in expansions.items():
            if disease in question_lower:
                enhanced_question += f" {expansion}"
        
        return enhanced_question
    
    def rerank_documents(self, docs: List[Document], question: str) -> List[Document]:
        """Rerank documents by relevance"""
        if not docs:
            return docs
        
        question_tokens = set(question.lower().split())
        scored_docs = []
        
        for doc in docs:
            content_lower = doc.page_content.lower()
            
            # Token overlap score
            content_tokens = set(content_lower.split())
            overlap = len(question_tokens.intersection(content_tokens))
            overlap_score = overlap / max(len(question_tokens), 1)
            
            # Keyword presence boost
            keyword_boost = 0
            important_keywords = ['disease', 'symptom', 'treatment', 'management', 'control', 'pathogen']
            for keyword in important_keywords:
                if keyword in content_lower:
                    keyword_boost += 0.05
            
            # Disease name exact match boost
            for token in question_tokens:
                if len(token) > 4 and token in content_lower:
                    keyword_boost += 0.1
            
            final_score = overlap_score + keyword_boost
            scored_docs.append((doc, final_score))
        
        # Sort by relevance and return top documents
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:8]]  # Return top 8
    
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        """
        Main retrieval method - required by BaseRetriever.
        Note: The argument name must be 'query' to match BaseRetriever signature.
        """
        try:
            # Preprocess the query
            enhanced_question = self.preprocess_query(query)
            
            # Retrieve documents using hybrid search if available
            if self._bm25_retriever:
                docs = self._ensemble_retriever.invoke(enhanced_question)
            else:
                docs = self._semantic_retriever.invoke(enhanced_question)
            
            # Also try original question if enhanced didn't work well
            if len(docs) < 5:
                additional_docs = self._semantic_retriever.invoke(query)
                # Combine and deduplicate
                all_docs = docs + additional_docs
                seen_content = set()
                unique_docs = []
                for doc in all_docs:
                    content_hash = hash(doc.page_content[:100])
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        unique_docs.append(doc)
                docs = unique_docs
            
            # Rerank for better relevance
            reranked_docs = self.rerank_documents(docs, query)
            
            return reranked_docs
            
        except Exception as e:
            print(f"Enhanced retrieval failed, using fallback: {e}")
            return self._semantic_retriever.invoke(query)

    # Note: Do not override 'invoke' manually. BaseRetriever handles it and calls _get_relevant_documents.

def load_retriever_from_disk():
    """Load the enhanced retriever with fallback to basic retriever"""
    try:
        # Pass path as a kwarg if needed, or rely on default
        enhanced_retriever = EnhancedRetriever(faiss_index_path=FAISS_INDEX_PATH)
        print("Enhanced retriever loaded successfully")
        return enhanced_retriever
    except Exception as e:
        print(f"Enhanced retriever failed, using basic retriever: {e}")
        # Fallback to basic retriever
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        return vector_store.as_retriever(search_kwargs={"k": 8})