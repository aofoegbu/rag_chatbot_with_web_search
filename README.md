# Ogelo RAG Chat Assistant

An advanced AI-powered document processing and knowledge retrieval system that provides comprehensive answers to any question with proper source citations.

https://ogeloragchat.live
 

![RAG Chatbot](screenshots/rag_chat.png)
 
## 📊 Features Overview

| Feature | Description | Status |
|---------|-------------|--------|
| Universal Q&A | Answer any question with sources | ✅ Active |
| PDF Processing | Extract text from PDF documents | ✅ Active |
| Image OCR | Extract text from images | ✅ Active |
| CSV Analysis | Process structured data | ✅ Active |
| Web Scraping | Extract content from URLs | ✅ Active |
| PostgreSQL | Primary database backend | ✅ Active |
| SQLite Fallback | Backup database system | ✅ Active |
| Conversation History | Track chat sessions | ✅ Active |
| Web Search | Real-time information access | ✅ Active |
| Source Citations | Proper attribution | ✅ Active |

     
## 🌟 Key Features

### Universal Question Answering
- **Ask Anything**: Get detailed answers to any question, from current events to technical concepts
- **Comprehensive Responses**: Every answer includes explanations, examples, and context
- **Source Citations**: All responses include up to 5 web sources with clear references
- **Real-time Information**: Access to current news, developments, and trending topics

### Multi-Format Document Support
- **PDF Processing**: Extract text and metadata from PDF documents
- **Image OCR**: Extract text from images (JPG, PNG, GIF, BMP, TIFF) using optical character recognition
- **CSV Analysis**: Process structured data with automatic summaries and statistics
- **Web Content**: Extract and process content from any URL
- **Text Files**: Support for plain text documents with multiple encoding formats

### Advanced RAG System
- **Intelligent Context Retrieval**: Combines document knowledge with web search results
- **Conversation History**: Maintains context across chat sessions
- **Embedding-based Search**: Uses semantic similarity for relevant content discovery
- **Multi-source Intelligence**: Integrates document content, web knowledge, and conversation history

### Database & Storage
- **Dual Database Support**: PostgreSQL (primary) with SQLite fallback
- **Auto-detection**: Automatically chooses the best available database
- **Conversation Logging**: All interactions stored for history and learning
- **Real-time Statistics**: Monitor system performance and usage

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- PostgreSQL (optional, SQLite used as fallback)
- Internet connection for web search functionality

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/aofoegbu/rag_chatbot_with_web_search
   cd ogelo-rag-chat-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit transformers torch sentence-transformers PyPDF2 numpy pandas pillow pytesseract requests beautifulsoup4 trafilatura psycopg2-binary
   ```

3. **Set up environment variables** (optional)
   ```bash
   export PERPLEXITY_API_KEY="your-api-key-here"
   export DATABASE_URL="postgresql://username:password@localhost/dbname"
   ```

4. **Run the application**
   ```bash
   streamlit run app.py --server.port 5000
   ```

### Quick Start

1. **Open your browser** to `http://localhost:5000`
2. **Ask any question** in the chat interface
3. **Upload documents** using the sidebar for additional context
4. **View responses** with comprehensive answers and source citations

## 📖 Usage Guide

### Asking Questions

The system can answer any type of question:

**Current Events:**
- "What are the latest AI developments?"
- "What happened in tech news today?"

**General Knowledge:**
- "How does photosynthesis work?"
- "Explain quantum computing"
- "What is machine learning?"

**Technical Concepts:**
- "How do neural networks function?"
- "Explain blockchain technology"

### Document Upload

1. **Click "Browse files"** in the sidebar
2. **Select your documents** (PDF, images, CSV, text files)
3. **Wait for processing** - you'll see confirmation when complete
4. **Ask questions** about your documents

### Web Content Processing

1. **Enter a URL** in the "Process URL" section
2. **Click "Process URL"** to extract content
3. **Ask questions** about the web content

## 🔧 System Architecture

### Core Components

- **Model Handler**: Manages AI models and response generation
- **Document Processor**: Extracts text from various file formats
- **RAG System**: Handles embedding generation and similarity search
- **Database Manager**: Manages data storage and retrieval
- **Web Search Integration**: Provides real-time information via Perplexity API

### Data Flow

1. **Input Processing**: Documents/URLs are processed and text extracted
2. **Chunking**: Text is split into semantic chunks for better retrieval
3. **Embedding**: Content is converted to vector embeddings
4. **Storage**: Chunks and embeddings stored in database
5. **Query Processing**: User questions trigger similarity search
6. **Response Generation**: AI combines document context with web search results
7. **Citation**: Sources are properly attributed in responses

## 🛠 Configuration

### Environment Variables

- `PERPLEXITY_API_KEY`: API key for web search functionality
- `DATABASE_URL`: PostgreSQL connection string (optional)
- `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`: PostgreSQL settings

### Model Configuration

The system supports multiple AI models:
- **Phi-3 Mini**: Recommended for general use (efficient, 4K context)
- **DialoGPT**: Alternative conversational model
- **Fallback System**: Enhanced rule-based responses when models unavailable

## 🧪 Testing

The system includes a comprehensive testing suite:

```bash
# Run through the Streamlit interface
# Navigate to the "System Testing" section in the sidebar
# Click "Run Comprehensive Tests"
```

Test coverage includes:
- Module imports and dependencies
- Database operations
- Document processing capabilities
- RAG system functionality
- Model handler operations
- File processing features
- Web integration

## 📈 Performance

### Optimizations
- **CPU-optimized models**: Efficient inference without GPU requirements
- **Lightweight embeddings**: Fast similarity computations
- **Chunked processing**: Handles large documents efficiently
- **Database indexing**: Quick retrieval of relevant content

### Resource Usage
- **Memory**: Moderate usage with model caching
- **Storage**: Efficient embedding storage with compression
- **Network**: Minimal usage except for web search queries

## 🔍 Troubleshooting

### Common Issues

**Web search not working:**
- Ensure `PERPLEXITY_API_KEY` is set
- Check internet connectivity
- Verify API key validity

**Database connection failed:**
- PostgreSQL: Check connection string and server status
- System automatically falls back to SQLite

**Document processing errors:**
- Ensure file format is supported
- Check file permissions and corruption
- Large files may take longer to process

**Model loading issues:**
- Models download automatically on first use
- Ensure sufficient disk space
- Check internet connection for downloads

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgments

- **Streamlit**: Web interface framework
- **Transformers**: AI model handling
- **Sentence Transformers**: Text embedding generation
- **PyPDF2**: PDF text extraction
- **Pytesseract**: OCR functionality
- **Perplexity AI**: Web search capabilities
- **PostgreSQL**: Primary database backend

## 📞 Support

For issues, questions, or contributions:
1. Check the troubleshooting section
2. Review existing issues
3. Create a new issue with detailed information
4. Join the community discussions

---

**Ogelo RAG Chat Assistant** - Comprehensive AI-powered knowledge retrieval with universal question answering capabilities.
