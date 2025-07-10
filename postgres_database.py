import os
import psycopg2
import numpy as np
import pickle
from typing import List, Tuple, Optional
import logging

class PostgreSQLManager:
    def __init__(self):
        """Initialize PostgreSQL database manager."""
        self.connection_string = os.getenv('DATABASE_URL')
        if not self.connection_string:
            raise Exception("PostgreSQL DATABASE_URL not found in environment variables")
        self.init_database()
    
    def get_connection(self):
        """Get a database connection."""
        return psycopg2.connect(self.connection_string)
    
    def init_database(self):
        """Initialize the database with required tables."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Create documents table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    embedding BYTEA NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create conversations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    user_message TEXT NOT NULL,
                    assistant_response TEXT NOT NULL,
                    context_used TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create index on filename for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_documents_filename 
                ON documents(filename)
            ''')
            
            # Create index on created_at for faster conversation retrieval
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_conversations_created_at 
                ON conversations(created_at DESC)
            ''')
            
            conn.commit()
            conn.close()
            print("PostgreSQL database initialized successfully!")
            
        except Exception as e:
            print(f"Error initializing PostgreSQL database: {e}")
            raise
    
    def store_document_chunk(self, filename: str, content: str, chunk_index: int, embedding: np.ndarray) -> bool:
        """Store a document chunk with its embedding."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Serialize embedding
            embedding_blob = pickle.dumps(embedding)
            
            cursor.execute('''
                INSERT INTO documents (filename, content, chunk_index, embedding)
                VALUES (%s, %s, %s, %s)
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
            conn = self.get_connection()
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
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO conversations (user_message, assistant_response, context_used)
                VALUES (%s, %s, %s)
            ''', (user_message, assistant_response, context_used))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error storing conversation: {e}")
            return False
    
    def get_document_count(self) -> int:
        """Get the total number of unique documents."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(DISTINCT filename) FROM documents')
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            print(f"Error getting document count: {e}")
            return 0
    
    def get_total_chunks(self) -> int:
        """Get the total number of document chunks."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM documents')
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            print(f"Error getting chunk count: {e}")
            return 0
    
    def clear_documents(self) -> bool:
        """Clear all documents from the database."""
        try:
            conn = self.get_connection()
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
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_message, assistant_response 
                FROM conversations 
                ORDER BY created_at DESC 
                LIMIT %s
            ''', (limit,))
            
            results = cursor.fetchall()
            conn.close()
            return results
        except Exception as e:
            print(f"Error getting recent conversations: {e}")
            return []
    
    def get_database_stats(self) -> dict:
        """Get comprehensive database statistics."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Get document stats
            cursor.execute('SELECT COUNT(DISTINCT filename) FROM documents')
            unique_docs = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM documents')
            total_chunks = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM conversations')
            total_conversations = cursor.fetchone()[0]
            
            # Get recent activity
            cursor.execute('''
                SELECT MAX(created_at) FROM documents
            ''')
            last_document = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT MAX(created_at) FROM conversations
            ''')
            last_conversation = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'unique_documents': unique_docs,
                'total_chunks': total_chunks,
                'total_conversations': total_conversations,
                'last_document_upload': last_document,
                'last_conversation': last_conversation,
                'database_type': 'PostgreSQL'
            }
            
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {}