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
    from document_processor import PDF_AVAILABLE, PANDAS_AVAILABLE
    file_types = ["txt"]
    help_parts = ["text files"]
    
    if PDF_AVAILABLE:
        file_types.append("pdf")
        help_parts.append("PDF")
    else:
        st.sidebar.info("📄 PDF support requires PyPDF2")
        
    if PANDAS_AVAILABLE:
        file_types.append("csv")
        help_parts.append("CSV")
    else:
        st.sidebar.info("📊 CSV support requires pandas")
    
    help_text = f"Upload {', '.join(help_parts)} files to add to the knowledge base"
except:
    file_types = ["txt"]
    help_text = "Upload text files to add to the knowledge base"

uploaded_file = st.sidebar.file_uploader(
    "Choose a document",
    type=file_types,
    help=help_text
)

if uploaded_file is not None:
    if st.sidebar.button("Process Document"):
        with st.sidebar.spinner("Processing document..."):
            try:
                success = st.session_state.doc_processor.process_document(uploaded_file)
                if success:
                    st.sidebar.success("Document processed successfully!")
                else:
                    st.sidebar.error("Failed to process document.")
            except Exception as e:
                st.sidebar.error(f"Error processing document: {str(e)}")

# Display document count
doc_count = st.session_state.db_manager.get_document_count()
st.sidebar.metric("Documents in Knowledge Base", doc_count)

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
                
                # Generate response with context
                response = st.session_state.model_handler.generate_response(prompt, context)
                
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

# Check model status
try:
    from model_handler import TRANSFORMERS_AVAILABLE
    from rag_system import SENTENCE_TRANSFORMERS_AVAILABLE
    
    if TRANSFORMERS_AVAILABLE:
        st.sidebar.markdown("🟢 **Model:** Phi-3-mini (Advanced)")
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
        
        # Test model loading
        try:
            test_response = st.session_state.model_handler.generate_response("Hello")
            if test_response:
                st.sidebar.success("✅ Model: Working")
            else:
                st.sidebar.warning("⚠️ Model: Limited functionality")
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
        
        # Test RAG system
        try:
            test_context, test_sources = st.session_state.rag_system.get_relevant_context("test query")
            st.sidebar.success("✅ RAG System: Working")
        except Exception as e:
            st.sidebar.error(f"❌ RAG System: {e}")

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
