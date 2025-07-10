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

# Check if PDF processing is available
try:
    from document_processor import PDF_AVAILABLE
    if PDF_AVAILABLE:
        file_types = ["pdf", "txt"]
        help_text = "Upload PDF or text files to add to the knowledge base"
    else:
        file_types = ["txt"]
        help_text = "PDF support not available. Please upload text files only."
        st.sidebar.info("📄 PDF support requires PyPDF2. Currently only text files are supported.")
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

st.sidebar.markdown("💾 **Database:** SQLite")
