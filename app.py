import streamlit as st
import os
from database import DatabaseManager
from model_handler import ModelHandler
from rag_system import RAGSystem
from document_processor import DocumentProcessor
import time

# Page configuration
st.set_page_config(
    page_title="Ogelo RAG Chat Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "db_manager" not in st.session_state:
    st.session_state.db_manager = DatabaseManager()
if "model_handler" not in st.session_state:
    st.session_state.model_handler = ModelHandler()
if "rag_system" not in st.session_state:
    st.session_state.rag_system = RAGSystem(st.session_state.db_manager)
if "doc_processor" not in st.session_state:
    st.session_state.doc_processor = DocumentProcessor(st.session_state.db_manager)

# Sidebar for document upload and management
st.sidebar.title("📚 Knowledge Base")

# Document upload section
st.sidebar.header("Upload Documents")

# Check available file processing capabilities
try:
    from document_processor import PDF_AVAILABLE, PANDAS_AVAILABLE, OCR_AVAILABLE, WEB_SCRAPING_AVAILABLE
    file_types = ["txt"]
    help_parts = ["text files"]
    
    if PDF_AVAILABLE:
        file_types.extend(["pdf"])
        help_parts.append("PDF")
    else:
        st.sidebar.info("📄 PDF support requires PyPDF2")
        
    if PANDAS_AVAILABLE:
        file_types.extend(["csv"])
        help_parts.append("CSV")
    else:
        st.sidebar.info("📊 CSV support requires pandas")
        
    if OCR_AVAILABLE:
        file_types.extend(["jpg", "jpeg", "png", "gif", "bmp", "tiff"])
        help_parts.append("images (OCR)")
    else:
        st.sidebar.info("🖼️ Image OCR requires tesseract and pytesseract")
    
    help_text = f"Upload {', '.join(help_parts)} to add to the knowledge base"
except:
    file_types = ["txt"]
    help_text = "Upload text files to add to the knowledge base"

uploaded_file = st.sidebar.file_uploader(
    "Choose a document",
    type=file_types,
    help=help_text
)

if uploaded_file is not None:
    # Show file info
    st.sidebar.write(f"**File:** {uploaded_file.name}")
    st.sidebar.write(f"**Size:** {uploaded_file.size} bytes")
    st.sidebar.write(f"**Type:** {uploaded_file.type}")
    
    if st.sidebar.button("Process Document"):
        with st.sidebar.spinner("Processing document..."):
            try:
                success = st.session_state.doc_processor.process_document(uploaded_file)
                if success:
                    st.sidebar.success("Document processed successfully!")
                    # Show processing details
                    doc_count = st.session_state.db_manager.get_document_count()
                    st.sidebar.info(f"Knowledge base now contains {doc_count} documents")
                else:
                    st.sidebar.error("Failed to process document. Check the console for details.")
            except Exception as e:
                st.sidebar.error(f"Error processing document: {str(e)}")
                # Also print to console for debugging
                print(f"Document processing error: {str(e)}")
                import traceback
                traceback.print_exc()

# Display document count
doc_count = st.session_state.db_manager.get_document_count()
st.sidebar.metric("Documents in Knowledge Base", doc_count)

# URL Processing Section
st.sidebar.markdown("---")
st.sidebar.header("Process URL Content")

url_input = st.sidebar.text_input(
    "Enter URL to extract content:",
    placeholder="https://example.com/article",
    help="Extract and process text content from web pages"
)

if url_input and st.sidebar.button("Process URL"):
    if WEB_SCRAPING_AVAILABLE:
        with st.sidebar.spinner("Fetching and processing URL content..."):
            try:
                success = st.session_state.doc_processor.process_url(url_input)
                if success:
                    st.sidebar.success("URL content processed successfully!")
                    doc_count = st.session_state.db_manager.get_document_count()
                    st.sidebar.info(f"Knowledge base now contains {doc_count} documents")
                else:
                    st.sidebar.error("Failed to process URL content. Check the console for details.")
            except Exception as e:
                st.sidebar.error(f"Error processing URL: {str(e)}")
                print(f"URL processing error: {str(e)}")
    else:
        st.sidebar.error("Web scraping is not available. Please install required packages.")

# Clear knowledge base button
if st.sidebar.button("Clear Knowledge Base", type="secondary"):
    if st.sidebar.confirm("Are you sure you want to clear all documents?"):
        st.session_state.db_manager.clear_documents()
        st.sidebar.success("Knowledge base cleared!")
        st.rerun()

# Main chat interface
st.title("🤖 Ogelo RAG Chat Assistant")
st.markdown("Chat with an AI assistant that can reference your uploaded documents.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📖 Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}:** {source}")

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Get relevant context using RAG
                context, sources = st.session_state.rag_system.get_relevant_context(prompt)
                
                # Generate response with context and conversation history
                response = st.session_state.model_handler.generate_response(
                    prompt, 
                    context, 
                    conversation_history=st.session_state.messages
                )
                
                # Store conversation in database
                st.session_state.db_manager.store_conversation(prompt, response, context)
                
                # Display response
                st.markdown(response)
                
                # Add assistant message to chat history
                assistant_message = {
                    "role": "assistant", 
                    "content": response,
                    "sources": sources if sources else []
                }
                st.session_state.messages.append(assistant_message)
                
                # Display sources if available
                if sources:
                    with st.expander("📖 Sources"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Source {i}:** {source}")
                            
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Clear chat button
if st.button("Clear Chat History", type="secondary"):
    st.session_state.messages = []
    st.rerun()

# Footer with model info and status
st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")

# Model Selection Section
st.sidebar.markdown("### 🤖 AI Model Selection")

# Get available models
try:
    available_models = st.session_state.model_handler.get_available_models()
    model_options = [f"{info['name']} - {info['description']}" for model_name, info in available_models.items()]
    model_keys = list(available_models.keys())
    
    # Find current model index
    current_model_index = 0
    try:
        current_model_index = model_keys.index(st.session_state.model_handler.model_name)
    except:
        pass
    
    selected_model_index = st.sidebar.selectbox(
        "Choose AI Model:",
        range(len(model_options)),
        index=current_model_index,
        format_func=lambda x: model_options[x],
        help="Select which AI model to use for generating responses"
    )
    
    # Switch model if changed
    selected_model_key = model_keys[selected_model_index]
    if selected_model_key != st.session_state.model_handler.model_name:
        with st.sidebar.spinner("Switching model..."):
            success = st.session_state.model_handler.switch_model(selected_model_key)
            if success:
                st.sidebar.success(f"Switched to {available_models[selected_model_key]['name']}")
                st.rerun()
            else:
                st.sidebar.error("Failed to switch model")
                
except Exception as e:
    st.sidebar.error(f"Model selection error: {e}")

# Check model status
try:
    from model_handler import TRANSFORMERS_AVAILABLE
    from rag_system import SENTENCE_TRANSFORMERS_AVAILABLE
    
    current_model_info = st.session_state.model_handler.get_available_models().get(
        st.session_state.model_handler.model_name, 
        {"name": "Unknown", "description": ""}
    )
    
    if TRANSFORMERS_AVAILABLE:
        st.sidebar.markdown(f"🟢 **Model:** {current_model_info['name']} (Advanced)")
    else:
        st.sidebar.markdown("🟡 **Model:** Rule-based (Simple)")
    
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        st.sidebar.markdown("🟢 **Embeddings:** all-MiniLM-L6-v2")
    else:
        st.sidebar.markdown("🟡 **Embeddings:** Text-based (Simple)")
        
except:
    st.sidebar.markdown("🟡 **Model:** Basic mode")
    st.sidebar.markdown("🟡 **Embeddings:** Simple matching")

db_type = st.session_state.db_manager.get_database_type()
st.sidebar.markdown(f"💾 **Database:** {db_type}")

# Check web search status
try:
    if st.session_state.model_handler.is_web_search_available():
        st.sidebar.markdown("🌐 **Web Search:** Enabled (Perplexity)")
    else:
        st.sidebar.markdown("🟡 **Web Search:** Disabled (API key needed)")
except:
    st.sidebar.markdown("🟡 **Web Search:** Not available")

# Add database statistics
if st.sidebar.button("📊 Show Database Stats"):
    stats = st.session_state.db_manager.get_database_stats()
    if stats:
        st.sidebar.markdown("### Database Statistics")
        st.sidebar.metric("Documents", stats.get('unique_documents', 0))
        st.sidebar.metric("Text Chunks", stats.get('total_chunks', 0))
        st.sidebar.metric("Conversations", stats.get('total_conversations', 0))

# Add comprehensive testing section
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 System Testing")

if st.sidebar.button("🔍 Test All Features"):
    with st.sidebar.expander("Test Results", expanded=True):
        # Test database connection
        try:
            test_count = st.session_state.db_manager.get_document_count()
            st.sidebar.success(f"✅ Database: Connected ({test_count} docs)")
        except Exception as e:
            st.sidebar.error(f"❌ Database: {e}")
        
        # Test model loading and selection
        try:
            available_models = st.session_state.model_handler.get_available_models()
            current_model = st.session_state.model_handler.model_name
            model_info = available_models.get(current_model, {})
            st.sidebar.success(f"✅ Model: {model_info.get('name', 'Unknown')} loaded")
            
            test_response = st.session_state.model_handler.generate_response("Hello")
            if test_response:
                st.sidebar.success("✅ Response Generation: Working")
            else:
                st.sidebar.warning("⚠️ Response Generation: Limited functionality")
        except Exception as e:
            st.sidebar.error(f"❌ Model: {e}")
        
        # Test embedding system
        try:
            test_embedding = st.session_state.rag_system.get_embedding("test")
            if test_embedding is not None and len(test_embedding) > 0:
                st.sidebar.success("✅ Embeddings: Working")
            else:
                st.sidebar.warning("⚠️ Embeddings: Basic mode")
        except Exception as e:
            st.sidebar.error(f"❌ Embeddings: {e}")
        
        # Test RAG system with knowledge integration
        try:
            test_context, test_sources = st.session_state.rag_system.get_relevant_context("machine learning")
            st.sidebar.success(f"✅ RAG System: Working ({len(test_sources)} sources)")
        except Exception as e:
            st.sidebar.error(f"❌ RAG System: {e}")
            
        # Test document processing capabilities
        try:
            from document_processor import PDF_AVAILABLE, PANDAS_AVAILABLE, OCR_AVAILABLE, WEB_SCRAPING_AVAILABLE
            
            capabilities = []
            if PDF_AVAILABLE: capabilities.append("PDF")
            if PANDAS_AVAILABLE: capabilities.append("CSV") 
            if OCR_AVAILABLE: capabilities.append("OCR")
            if WEB_SCRAPING_AVAILABLE: capabilities.append("Web")
            
            st.sidebar.success(f"✅ Document Processing: {', '.join(capabilities)}")
        except Exception as e:
            st.sidebar.error(f"❌ Document Processing: {e}")
            
        # Test web search integration
        try:
            if st.session_state.model_handler.is_web_search_available():
                success, message = st.session_state.model_handler.test_web_search()
                if success:
                    st.sidebar.success("✅ Web Search: Connected")
                else:
                    st.sidebar.warning(f"⚠️ Web Search: {message}")
            else:
                st.sidebar.info("ℹ️ Web Search: API key needed")
        except Exception as e:
            st.sidebar.error(f"❌ Web Search: {e}")
            
        # Test knowledge integration
        try:
            from web_search_integration import WebSearchIntegrator
            integrator = WebSearchIntegrator()
            enhanced_context = integrator._enhance_with_knowledge("test query")
            st.sidebar.success("✅ Knowledge Integration: Working")
        except Exception as e:
            st.sidebar.error(f"❌ Knowledge Integration: {e}")

# Run comprehensive test suite button
if st.sidebar.button("🔬 Run Full Test Suite"):
    with st.sidebar.spinner("Running comprehensive tests..."):
        try:
            import subprocess
            result = subprocess.run(['python', 'comprehensive_test.py'], 
                                  capture_output=True, text=True, cwd='.')
            
            with st.sidebar.expander("Full Test Results", expanded=True):
                if result.returncode == 0:
                    st.sidebar.success("✅ All tests completed successfully")
                else:
                    st.sidebar.warning("⚠️ Some tests had issues")
                
                # Show test output
                if result.stdout:
                    st.sidebar.text_area("Test Output", result.stdout, height=300)
                if result.stderr:
                    st.sidebar.text_area("Test Errors", result.stderr, height=150)
                    
        except Exception as e:
            st.sidebar.error(f"❌ Test Suite Error: {e}")

# Add conversation history viewer and integration
if st.sidebar.button("📜 View Recent Conversations"):
    recent_conversations = st.session_state.db_manager.get_recent_conversations(5)
    if recent_conversations:
        with st.sidebar.expander("Recent Conversations", expanded=True):
            for i, (user_msg, assistant_msg) in enumerate(recent_conversations, 1):
                st.sidebar.markdown(f"**{i}. User:** {user_msg[:50]}...")
                st.sidebar.markdown(f"**Assistant:** {assistant_msg[:50]}...")
                st.sidebar.markdown("---")
    else:
        st.sidebar.info("No conversations found in database")

# Add option to load previous conversations into current chat
st.sidebar.markdown("---")
st.sidebar.markdown("### 💬 Previous Conversations")

if st.sidebar.button("🔄 Load Previous Context"):
    recent_conversations = st.session_state.db_manager.get_recent_conversations(3)
    if recent_conversations:
        # Add recent conversations to current chat session
        for user_msg, assistant_msg in reversed(recent_conversations):
            # Only add if not already in current session
            if not any(msg["content"] == user_msg for msg in st.session_state.messages if msg["role"] == "user"):
                st.session_state.messages.insert(0, {"role": "user", "content": user_msg})
                st.session_state.messages.insert(1, {"role": "assistant", "content": assistant_msg})
        st.sidebar.success(f"Loaded {len(recent_conversations)} previous conversations")
        st.rerun()
    else:
        st.sidebar.info("No previous conversations to load")
