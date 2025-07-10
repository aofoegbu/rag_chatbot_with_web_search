try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

import io
from typing import List
from database import DatabaseManager
from rag_system import RAGSystem

class DocumentProcessor:
    def __init__(self, db_manager: DatabaseManager):
        """Initialize document processor with database manager."""
        self.db_manager = db_manager
        self.rag_system = RAGSystem(db_manager)
        self.chunk_size = 500  # Characters per chunk
        self.chunk_overlap = 50  # Overlap between chunks
    
    def process_document(self, uploaded_file) -> bool:
        """Process an uploaded document and store it in the database."""
        try:
            filename = uploaded_file.name
            file_extension = filename.lower().split('.')[-1]
            
            # Extract text based on file type
            if file_extension == 'pdf':
                text = self._extract_text_from_pdf(uploaded_file)
            elif file_extension == 'txt':
                text = self._extract_text_from_txt(uploaded_file)
            elif file_extension == 'csv':
                text = self._extract_text_from_csv(uploaded_file)
            else:
                print(f"Unsupported file type: {file_extension}")
                return False
            
            if not text.strip():
                print("No text content found in the document")
                return False
            
            # Split text into chunks
            chunks = self._split_text_into_chunks(text)
            
            # Process and store each chunk
            for i, chunk in enumerate(chunks):
                # Get embedding for the chunk
                embedding = self.rag_system.get_embedding(chunk)
                
                # Store chunk in database
                success = self.db_manager.store_document_chunk(
                    filename, chunk, i, embedding
                )
                
                if not success:
                    print(f"Failed to store chunk {i} of {filename}")
                    return False
            
            print(f"Successfully processed {filename} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            print(f"Error processing document: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_text_from_pdf(self, uploaded_file) -> str:
        """Extract text from a PDF file."""
        if not PDF_AVAILABLE:
            return "PDF processing is not available. Please install PyPDF2 or upload a text file instead."
            
        try:
            # Reset file pointer and read the PDF file
            uploaded_file.seek(0)
            pdf_content = uploaded_file.read()
            
            # Create PDF reader from bytes
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            
            text_parts = []
            
            # Extract text from each page
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        text_parts.append(f"Page {page_num + 1}:\n{page_text}")
                except Exception as page_error:
                    print(f"Error extracting text from page {page_num + 1}: {page_error}")
                    continue
            
            if not text_parts:
                return "No readable text found in the PDF. The PDF might be image-based or corrupted."
                
            # Add PDF metadata
            metadata_info = []
            if hasattr(pdf_reader, 'metadata') and pdf_reader.metadata:
                try:
                    if '/Title' in pdf_reader.metadata:
                        metadata_info.append(f"Title: {pdf_reader.metadata['/Title']}")
                    if '/Author' in pdf_reader.metadata:
                        metadata_info.append(f"Author: {pdf_reader.metadata['/Author']}")
                except:
                    pass  # Skip metadata if not accessible
            
            # Combine all text
            full_text = ""
            if metadata_info:
                full_text += "PDF Information:\n" + "\n".join(metadata_info) + "\n\n"
            
            full_text += "\n\n".join(text_parts)
            
            print(f"Successfully extracted text from PDF: {len(text_parts)} pages, {len(full_text)} characters")
            return full_text
            
        except Exception as e:
            error_msg = f"Error extracting text from PDF: {e}"
            print(error_msg)
            return f"Failed to process PDF file: {str(e)}. Please ensure the PDF is not corrupted or password-protected."
    
    def _extract_text_from_txt(self, uploaded_file) -> str:
        """Extract text from a text file."""
        try:
            # Reset file pointer and read content
            uploaded_file.seek(0)
            content = uploaded_file.read()
            
            # Try to decode as UTF-8, fallback to latin-1 if that fails
            if isinstance(content, bytes):
                try:
                    text = content.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        text = content.decode('latin-1')
                    except UnicodeDecodeError:
                        text = content.decode('utf-8', errors='ignore')
            else:
                text = content
                
            if not text.strip():
                return "The text file appears to be empty."
                
            print(f"Successfully extracted text from TXT: {len(text)} characters")
            return text
            
        except Exception as e:
            error_msg = f"Error reading text file: {e}"
            print(error_msg)
            return f"Failed to process text file: {str(e)}"
    
    def _extract_text_from_csv(self, uploaded_file) -> str:
        """Extract text from a CSV file."""
        if not PANDAS_AVAILABLE:
            return "CSV processing is not available. Please install pandas or upload a text file instead."
            
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            
            # Convert DataFrame to a readable text format
            text_parts = []
            
            # Add column headers
            text_parts.append("CSV Data Structure:")
            text_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
            text_parts.append(f"Total rows: {len(df)}")
            text_parts.append("\nData Content:")
            
            # Add each row as structured text
            for index, row in df.iterrows():
                row_text = f"Row {index + 1}: "
                row_items = []
                for col in df.columns:
                    value = str(row[col]) if pd.notna(row[col]) else "N/A"
                    row_items.append(f"{col}: {value}")
                row_text += ", ".join(row_items)
                text_parts.append(row_text)
                
                # Limit to first 100 rows to avoid memory issues
                if index >= 99:
                    text_parts.append(f"... and {len(df) - 100} more rows")
                    break
            
            # Add summary statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                text_parts.append("\nNumeric Summary:")
                for col in numeric_cols:
                    stats = df[col].describe()
                    text_parts.append(f"{col}: mean={stats['mean']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")
            
            return "\n".join(text_parts)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return f"Error processing CSV file: {str(e)}"
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to break at a sentence or word boundary
            if end < len(text):
                # Look for sentence endings
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start:
                    end = sentence_end + 1
                else:
                    # Look for word boundaries
                    word_end = text.rfind(' ', start, end)
                    if word_end > start:
                        end = word_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
            # Ensure we don't go backwards
            if start <= 0:
                start = end
        
        return chunks
