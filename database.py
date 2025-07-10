import sqlite3
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional

class DatabaseManager:
    def __init__(self, db_path: str = "rag_database.db"):
        """Initialize database manager with SQLite database."""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_message TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                context_used TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_document_chunk(self, filename: str, content: str, chunk_index: int, embedding: np.ndarray) -> bool:
        """Store a document chunk with its embedding."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Serialize embedding
            embedding_blob = pickle.dumps(embedding)
            
            cursor.execute('''
                INSERT INTO documents (filename, content, chunk_index, embedding)
                VALUES (?, ?, ?, ?)
            ''', (filename, content, chunk_index, embedding_blob))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error storing document chunk: {e}")
            return False
    
    def search_similar_chunks(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Search for similar document chunks using cosine similarity."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT filename, content, embedding FROM documents')
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return []
            
            similarities = []
            for filename, content, embedding_blob in results:
                # Deserialize embedding
                embedding = pickle.loads(embedding_blob)
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, embedding)
                similarities.append((filename, content, similarity))
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[2], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            print(f"Error searching similar chunks: {e}")
            return []
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def store_conversation(self, user_message: str, assistant_response: str, context_used: str = None) -> bool:
        """Store a conversation exchange."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO conversations (user_message, assistant_response, context_used)
                VALUES (?, ?, ?)
            ''', (user_message, assistant_response, context_used))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error storing conversation: {e}")
            return False
    
    def get_document_count(self) -> int:
        """Get the total number of document chunks."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(DISTINCT filename) FROM documents')
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            print(f"Error getting document count: {e}")
            return 0
    
    def clear_documents(self) -> bool:
        """Clear all documents from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM documents')
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error clearing documents: {e}")
            return False
    
    def get_recent_conversations(self, limit: int = 10) -> List[Tuple[str, str]]:
        """Get recent conversations."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_message, assistant_response 
                FROM conversations 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            
            results = cursor.fetchall()
            conn.close()
            return results
        except Exception as e:
            print(f"Error getting recent conversations: {e}")
            return []
