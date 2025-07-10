try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    
import numpy as np
from typing import List, Tuple
from database import DatabaseManager

class RAGSystem:
    def __init__(self, db_manager: DatabaseManager):
        """Initialize RAG system with database manager."""
        self.db_manager = db_manager
        self.embedding_model = None
        self.load_embedding_model()
    
    def load_embedding_model(self):
        """Load the sentence transformer model for embeddings."""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                print("Loading embedding model...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Embedding model loaded successfully!")
            else:
                print("Sentence transformers not available - using simple text similarity")
                self.embedding_model = "simple"
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            print("Falling back to simple text similarity")
            self.embedding_model = "simple"
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a given text."""
        if self.embedding_model is None:
            raise Exception("Embedding model not loaded")
        
        try:
            if self.embedding_model == "simple":
                # Simple text-based embedding using character frequencies
                return self._simple_text_embedding(text)
            else:
                embedding = self.embedding_model.encode(text)
                return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return self._simple_text_embedding(text)  # Fallback to simple embedding
    
    def _simple_text_embedding(self, text: str) -> np.ndarray:
        """Create a simple embedding based on text characteristics."""
        # Normalize text
        text = text.lower()
        
        # Create a fixed-size vector based on text features
        embedding = np.zeros(384)  # Standard embedding size
        
        # Simple features
        words = text.split()
        if words:
            # Word count feature
            embedding[0] = min(len(words) / 100.0, 1.0)
            
            # Character count feature  
            embedding[1] = min(len(text) / 1000.0, 1.0)
            
            # Simple hash-based features for words
            for i, word in enumerate(words[:50]):  # Use first 50 words
                word_hash = hash(word) % 382  # Remaining dimensions
                embedding[word_hash + 2] += 1.0 / len(words)
        
        return embedding
    
    def get_relevant_context(self, query: str, top_k: int = 3) -> Tuple[str, List[str]]:
        """Get relevant context for a query using RAG."""
        try:
            # Get query embedding
            query_embedding = self.get_embedding(query)
            
            # Search for similar chunks
            similar_chunks = self.db_manager.search_similar_chunks(query_embedding, top_k)
            
            if not similar_chunks:
                return "", []
            
            # Extract context and sources
            context_parts = []
            sources = []
            
            for filename, content, similarity in similar_chunks:
                # Only include chunks with reasonable similarity
                if similarity > 0.3:  # Threshold for relevance
                    context_parts.append(content)
                    sources.append(f"{filename} (similarity: {similarity:.2f})")
            
            # Combine context
            context = "\n\n".join(context_parts)
            
            return context, sources
            
        except Exception as e:
            print(f"Error getting relevant context: {e}")
            return "", []
    
    def is_embedding_model_loaded(self) -> bool:
        """Check if the embedding model is loaded."""
        return self.embedding_model is not None
