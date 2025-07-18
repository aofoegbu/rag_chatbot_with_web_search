import sqlite3
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional

# Try to import PostgreSQL support
try:
    import psycopg2
    from postgres_database import PostgreSQLManager
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

class DatabaseManager:
    def __init__(self, db_path: str = "rag_database.db", use_postgres: bool = None):
        """Initialize database manager with SQLite or PostgreSQL."""
        # Auto-detect database type if not specified
        if use_postgres is None:
            use_postgres = POSTGRES_AVAILABLE and os.getenv('DATABASE_URL') is not None
        
        self.use_postgres = use_postgres
        
        if self.use_postgres and POSTGRES_AVAILABLE:
            try:
                self.postgres_manager = PostgreSQLManager()
                print("Using PostgreSQL database")
            except Exception as e:
                print(f"Failed to connect to PostgreSQL: {e}")
                print("Falling back to SQLite")
                self.use_postgres = False
                self.db_path = db_path
                self.init_database()
        else:
            self.use_postgres = False
            self.db_path = db_path
            self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables."""
        if self.use_postgres:
            return  # PostgreSQL initialization is handled in PostgreSQLManager
            
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
        if self.use_postgres:
            return self.postgres_manager.store_document_chunk(filename, content, chunk_index, embedding)
            
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
        if self.use_postgres:
            return self.postgres_manager.search_similar_chunks(query_embedding, top_k)
            
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
        if self.use_postgres:
            return self.postgres_manager.store_conversation(user_message, assistant_response, context_used)
            
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
        """Get the total number of unique documents."""
        if self.use_postgres:
            return self.postgres_manager.get_document_count()
            
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
        if self.use_postgres:
            return self.postgres_manager.clear_documents()
            
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
        if self.use_postgres:
            return self.postgres_manager.get_recent_conversations(limit)
            
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
    
    def get_database_type(self) -> str:
        """Get the type of database being used."""
        return "PostgreSQL" if self.use_postgres else "SQLite"
    
    def get_database_stats(self) -> dict:
        """Get comprehensive database statistics."""
        if self.use_postgres:
            return self.postgres_manager.get_database_stats()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get document stats
            cursor.execute('SELECT COUNT(DISTINCT filename) FROM documents')
            unique_docs = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM documents')
            total_chunks = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM conversations')
            total_conversations = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'unique_documents': unique_docs,
                'total_chunks': total_chunks,
                'total_conversations': total_conversations,
                'database_type': 'SQLite'
            }
            
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {}
