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
- **Added Web Search Integration (Latest)**: Implemented Perplexity API for real-time internet information when needed
- **Smart Search Logic**: System automatically determines when to use web search vs internal knowledge
- **Web Source Citations**: All web search results include proper source links and references
- **Fixed Answering Mechanism**: Resolved issue where model wasn't showing responses properly
- **Improved Response Formatting**: Clean, well-structured answers with clear headings and bullet points
- **Enhanced Knowledge Coverage**: 8 major domains with comprehensive, detailed responses plus web search
- **Added PostgreSQL Support**: Dual database system with auto-detection
- **Enhanced Testing Suite**: Comprehensive feature testing with visual feedback
- **Conversation Storage**: All interactions stored in database for history tracking
- **Database Statistics**: Real-time metrics display for system monitoring
- **Improved UI**: Better status indicators and system information display
- **CSV File Support**: Added pandas-based CSV processing with structured data extraction
- **Previous Context Integration**: Load and integrate conversation history into chat sessions
- **Enhanced RAG Context**: Include conversation history in response generation for continuity
- **Advanced Prompt Engineering**: Structured prompts with in-context examples and comprehensive knowledge integration
- **Multi-Source Intelligence**: Combines document knowledge, conversation history, general knowledge, and contextual examples
- **Knowledge Base Integration**: Enhanced responses with relevant internal knowledge and reasoning capabilities
- **Comprehensive Citation System**: Every answer includes proper references and source attributions
- **Knowledge Base References**: Detailed citations for internal knowledge, best practices, and domain expertise
- **Model Selection Interface**: User can choose between Phi-3 Mini and DialoGPT models with live switching
- **Comprehensive Testing Suite**: Automated testing system with 35+ test cases covering all components and features
- **Image OCR Support**: Added pytesseract integration for extracting text from images (JPG, PNG, GIF, BMP, TIFF)
- **URL Content Extraction**: Implemented web scraping with trafilatura and BeautifulSoup for processing website content
- **Multi-format Document Support**: Now supports PDF, text, CSV, images, and web URLs
- **Enhanced PDF Processing**: Improved metadata extraction and error handling for PDF files

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
- **Purpose**: Extracts and processes text from uploaded documents and web content
- **Supported Formats**: 
  - PDF files (via PyPDF2) with metadata extraction
  - Plain text files (.txt) with multi-encoding support
  - CSV files (via pandas) with structured data extraction and statistics
  - Image files (JPG, PNG, GIF, BMP, TIFF) with OCR text extraction via pytesseract
  - Web URLs with content extraction using trafilatura and BeautifulSoup fallback
- **Text Chunking**: Splits documents into 500-character chunks with 50-character overlap for better context retrieval
- **CSV Processing**: Converts tabular data to searchable text with row details and numeric summaries
- **OCR Processing**: Extracts text from images with metadata and error handling
- **Web Scraping**: Intelligent content extraction from websites with title and timestamp metadata

### 3. RAG System (`rag_system.py` + `web_search_integration.py`)
- **Purpose**: Handles embedding generation, similarity search, and knowledge enhancement for retrieval
- **Embedding Model**: Uses `all-MiniLM-L6-v2` from sentence-transformers for fast, lightweight embeddings with simple fallback
- **Retrieval Strategy**: Cosine similarity search with configurable top-k results and relevance thresholding
- **Context Integration**: Automatically includes recent conversation history for contextual continuity
- **Knowledge Enhancement**: Integrates general knowledge, examples, and reasoning with document context
- **Multi-Source Context**: Combines document chunks, conversation history, and web knowledge for comprehensive responses

### 4. Model Handler (`model_handler.py`)
- **Purpose**: Manages multiple language models with user selection and advanced prompt engineering
- **Available Models**: 
  - Phi-3 Mini (Recommended): Microsoft's efficient 4K context model
  - DialoGPT Medium: Conversational AI model for dialogue
- **Model Switching**: Live model switching without restarting the application
- **Prompt Engineering**: Structured prompts with in-context examples, conversation history, and comprehensive context
- **Web Search Integration**: Real-time internet search via Perplexity API for current information
- **Smart Response Logic**: Automatically determines when to use web search vs internal knowledge
- **Knowledge Integration**: Combines document context with general knowledge, examples, reasoning, and web sources
- **Reference System**: Every response includes detailed citations and source attributions for transparency
- **Fallback Strategy**: Enhanced rule-based responses with knowledge integration when ML models unavailable
- **Optimization**: CPU-optimized loading with float32 precision and low memory usage

### 5. Main Application (`app.py`)
- **Purpose**: Streamlit interface orchestrating all components with advanced user controls
- **State Management**: Initializes and maintains all system components in session state
- **User Interface**: 
  - Model selection dropdown with live switching capabilities
  - Clean sidebar for document management and system monitoring
  - Main chat area with source citations and conversation continuity
  - Comprehensive testing suite with automated test runner
  - Real-time system status indicators showing current model and capabilities
  - Database statistics and conversation history viewer with integration options

### 6. Testing Suite (`comprehensive_test.py`)
- **Purpose**: Automated testing framework for validating all system components
- **Test Coverage**: 35+ test cases covering imports, database, document processing, RAG, models, files, and web integration
- **Test Categories**: Module imports, database operations, document capabilities, embedding systems, model functionality, file processing, web integration
- **Integration**: Embedded in Streamlit UI with detailed results display
- **Success Metrics**: Comprehensive reporting with pass/warning/fail statistics

## Data Flow

1. **Content Input**: User uploads documents (PDF, text, CSV, images) or provides URLs through Streamlit interface
2. **Text Extraction**: Document processor extracts and cleans text content using appropriate methods:
   - PDF: PyPDF2 with metadata extraction
   - Images: OCR with pytesseract
   - URLs: Web scraping with trafilatura/BeautifulSoup
   - CSV: Structured data conversion with pandas
3. **Chunking**: Text is split into overlapping chunks for better retrieval
4. **Embedding Generation**: Each chunk is converted to vector embedding using sentence transformer
5. **Storage**: Chunks and embeddings are stored in PostgreSQL or SQLite database
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
- **PyPDF2**: PDF text extraction with metadata support
- **numpy**: Numerical operations for embeddings
- **pillow**: Image processing and format conversion
- **pytesseract**: OCR text extraction from images
- **requests**: HTTP requests for web content fetching
- **beautifulsoup4**: HTML parsing and content extraction
- **trafilatura**: Advanced web content extraction

### Web Framework
- **streamlit**: Complete web interface framework

### Database
- **sqlite3**: Built-in Python SQLite interface  
- **psycopg2-binary**: PostgreSQL adapter for Python
- **pickle**: Serialization for embedding storage
- **pandas**: CSV data processing and analysis

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