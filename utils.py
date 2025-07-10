import os
import logging
from typing import Any, Dict

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('rag_chat.log'),
            logging.StreamHandler()
        ]
    )

def get_env_variable(var_name: str, default_value: Any = None) -> Any:
    """Get environment variable with fallback."""
    return os.getenv(var_name, default_value)

def validate_file_type(filename: str, allowed_extensions: list) -> bool:
    """Validate if file type is allowed."""
    if not filename:
        return False
    
    file_extension = filename.lower().split('.')[-1]
    return file_extension in allowed_extensions

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def format_similarity_score(score: float) -> str:
    """Format similarity score as percentage."""
    return f"{score * 100:.1f}%"

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove common artifacts from PDF extraction
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    
    return text.strip()

def chunk_text_by_sentences(text: str, max_chunk_size: int = 500) -> list:
    """Split text into chunks by sentences while respecting max size."""
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Add period back to sentence (except for the last one)
        if not sentence.endswith('.') and sentence != sentences[-1]:
            sentence += '.'
        
        # Check if adding this sentence would exceed the limit
        if len(current_chunk) + len(sentence) + 1 > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # If single sentence is too long, add it anyway
                chunks.append(sentence.strip())
                current_chunk = ""
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def get_model_info() -> Dict[str, str]:
    """Get information about the models being used."""
    return {
        "llm_model": "microsoft/Phi-3-mini-4k-instruct",
        "embedding_model": "all-MiniLM-L6-v2",
        "database": "SQLite",
        "framework": "Streamlit"
    }
