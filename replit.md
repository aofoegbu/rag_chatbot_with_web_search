# Ogelo RAG Chat Assistant

## Overview

Ogelo RAG Chat Assistant is an advanced Retrieval-Augmented Generation (RAG) chat system built with Streamlit. The application allows users to upload documents to build a knowledge base and ask questions about content. It features dual database support (PostgreSQL/SQLite), adaptive ML models with fallback systems, comprehensive testing tools, and conversation history tracking.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Web Interface**: Simple single-page application with a main chat interface and sidebar for document management
- **Session State Management**: Uses Streamlit's session state to maintain application state across user interactions
- **Real-time Processing**: Immediate feedback during document processing and chat interactions

### Backend Architecture
- **Modular Python Components**: Separated into distinct modules for database management, document processing, RAG operations, and model handling
- **SQLite Database**: Lightweight file-based database for storing document chunks, embeddings, and conversation history
- **CPU-Optimized Model Loading**: Uses quantized models with CPU optimization for running on resource-constrained environments

### Data Storage Solutions
- **Dual Database Support**: 
  - **PostgreSQL**: Primary choice with auto-detection, better for scalability
  - **SQLite**: Fallback option, file-based for simplicity
- **Database Tables**:
  - `documents`: Stores document chunks with embeddings and metadata
  - `conversations`: Stores chat history with context information for learning
- **BLOB Storage**: Embeddings serialized using pickle and stored as binary data
- **Auto-Detection**: System automatically chooses PostgreSQL if available, falls back to SQLite

## Recent Changes (July 2025)
- **Added PostgreSQL Support**: Dual database system with auto-detection
- **Enhanced Testing Suite**: Comprehensive feature testing with visual feedback
- **Conversation Storage**: All interactions stored in database for history tracking
- **Database Statistics**: Real-time metrics display for system monitoring
- **Improved UI**: Better status indicators and system information display

## Key Components

### 1. Database Manager (`database.py` + `postgres_database.py`)
- **Purpose**: Handles all database operations with dual-backend support
- **Key Features**: 
  - Auto-detection between PostgreSQL and SQLite
  - Document chunk storage with embeddings
  - Conversation logging and history retrieval
  - Comprehensive statistics and monitoring
  - Graceful fallback systems
- **Design Choice**: PostgreSQL for production, SQLite for development/fallback

### 2. Document Processor (`document_processor.py`)
- **Purpose**: Extracts and processes text from uploaded documents
- **Supported Formats**: PDF (via PyPDF2) and plain text files
- **Text Chunking**: Splits documents into 500-character chunks with 50-character overlap for better context retrieval

### 3. RAG System (`rag_system.py`)
- **Purpose**: Handles embedding generation and similarity search for retrieval
- **Embedding Model**: Uses `all-MiniLM-L6-v2` from sentence-transformers for fast, lightweight embeddings
- **Retrieval Strategy**: Cosine similarity search with configurable top-k results

### 4. Model Handler (`model_handler.py`)
- **Purpose**: Manages the language model for generating responses
- **Primary Model**: Microsoft Phi-3-mini-4k-instruct for efficient CPU-based inference
- **Fallback Strategy**: DialoGPT-medium as backup if primary model fails to load
- **Optimization**: CPU-optimized loading with float32 precision and low memory usage

### 5. Main Application (`app.py`)
- **Purpose**: Streamlit interface orchestrating all components
- **State Management**: Initializes and maintains all system components in session state
- **User Interface**: 
  - Clean sidebar for document management and system monitoring
  - Main chat area with source citations
  - Comprehensive testing suite integrated into sidebar
  - Real-time system status indicators
  - Database statistics and conversation history viewer

## Data Flow

1. **Document Upload**: User uploads PDF/text file through Streamlit interface
2. **Text Extraction**: Document processor extracts and cleans text content
3. **Chunking**: Text is split into overlapping chunks for better retrieval
4. **Embedding Generation**: Each chunk is converted to vector embedding using sentence transformer
5. **Storage**: Chunks and embeddings are stored in SQLite database
6. **Query Processing**: User questions are embedded and compared against stored chunks
7. **Context Retrieval**: Most similar chunks are retrieved as context
8. **Response Generation**: Language model generates response using retrieved context
9. **Conversation Logging**: All exchanges automatically stored in database with context
10. **System Monitoring**: Real-time testing and statistics available through UI

## External Dependencies

### Core ML Libraries
- **sentence-transformers**: For generating text embeddings
- **transformers**: For language model handling
- **torch**: PyTorch backend for model operations

### Document Processing
- **PyPDF2**: PDF text extraction
- **numpy**: Numerical operations for embeddings

### Web Framework
- **streamlit**: Complete web interface framework

### Database
- **sqlite3**: Built-in Python SQLite interface
- **pickle**: Serialization for embedding storage

## Deployment Strategy

### Local Development
- **Single File Database**: SQLite database file for easy portability
- **CPU Optimization**: Models configured for CPU-only inference
- **Resource Management**: Lightweight models chosen for limited hardware

### Scalability Considerations
- **Database**: SQLite suitable for single-user applications; would need migration to PostgreSQL for multi-user scenarios
- **Model Serving**: Currently loads models in-process; could be moved to separate service for better resource management
- **Storage**: File-based storage works for small-scale; would need cloud storage for larger deployments

### Configuration
- **Environment Variables**: Utility functions provided for environment-based configuration
- **Logging**: Structured logging setup with file and console output
- **Error Handling**: Graceful fallbacks for model loading and processing failures

The architecture prioritizes simplicity and local deployment while maintaining modularity for future enhancements. The choice of lightweight models and SQLite makes it suitable for personal use or small team deployments without requiring complex infrastructure.