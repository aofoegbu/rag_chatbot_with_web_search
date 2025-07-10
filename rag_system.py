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
    
    def get_relevant_context(self, query: str, top_k: int = 3, include_conversation_history: bool = True) -> Tuple[str, List[str]]:
        """Get relevant context for a query using RAG with optional conversation history."""
        try:
            # Get query embedding
            query_embedding = self.get_embedding(query)
            
            # Search for similar chunks
            similar_chunks = self.db_manager.search_similar_chunks(query_embedding, top_k)
            
            # Extract context and sources
            context_parts = []
            sources = []
            
            # Add document context
            for filename, content, similarity in similar_chunks:
                # Only include chunks with reasonable similarity
                if similarity > 0.3:  # Threshold for relevance
                    context_parts.append(f"From {filename}: {content}")
                    sources.append(f"{filename} (similarity: {similarity:.2f})")
            
            # Add conversation history context if requested and available
            if include_conversation_history:
                try:
                    recent_conversations = self.db_manager.get_recent_conversations(2)
                    if recent_conversations:
                        context_parts.append("\nRecent conversation context:")
                        for i, (user_msg, assistant_msg) in enumerate(recent_conversations):
                            # Truncate long messages for context
                            user_short = user_msg[:100] + "..." if len(user_msg) > 100 else user_msg
                            assistant_short = assistant_msg[:150] + "..." if len(assistant_msg) > 150 else assistant_msg
                            context_parts.append(f"Previous Q{i+1}: {user_short}")
                            context_parts.append(f"Previous A{i+1}: {assistant_short}")
                        sources.append("Recent conversations")
                except Exception as conv_error:
                    print(f"Could not retrieve conversation history: {conv_error}")
            
            # Enhance with web knowledge integration
            try:
                from web_search_integration import web_search_integrator
                base_context = "\n\n".join(context_parts) if context_parts else ""
                enhanced_context, web_sources = web_search_integrator.search_and_enhance(query, base_context)
                
                # Add web sources to the sources list
                if web_sources:
                    sources.extend(web_sources)
                
                return enhanced_context, sources
                
            except ImportError:
                # Fallback if web search integration is not available
                context = "\n\n".join(context_parts) if context_parts else "No relevant context found."
                return context, sources
            
        except Exception as e:
            print(f"Error getting relevant context: {e}")
            return "Error retrieving context.", []
    
    def is_embedding_model_loaded(self) -> bool:
        """Check if the embedding model is loaded."""
        return self.embedding_model is not None
