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

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import requests
    from bs4 import BeautifulSoup
    import trafilatura
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False

import io
import re
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
            elif file_extension in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']:
                text = self._extract_text_from_image(uploaded_file)
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
            
            print(f"Successfully extracted text from CSV: {len(df)} rows, {len(text_parts)} chunks")
            return "\n".join(text_parts)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return f"Error processing CSV file: {str(e)}"
    
    def _extract_text_from_image(self, uploaded_file) -> str:
        """Extract text from an image using OCR."""
        if not OCR_AVAILABLE:
            return "OCR processing is not available. Please install Pillow and pytesseract or upload a text file instead."
            
        try:
            # Reset file pointer and read image
            uploaded_file.seek(0)
            image_data = uploaded_file.read()
            
            # Open image with PIL
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract text using OCR
            extracted_text = pytesseract.image_to_string(image)
            
            if not extracted_text.strip():
                return "No readable text found in the image. The image might not contain text or the text might not be clear enough for OCR."
            
            # Add image metadata
            image_info = [
                f"Image Information:",
                f"Format: {image.format}",
                f"Size: {image.size[0]}x{image.size[1]} pixels",
                f"Mode: {image.mode}",
                "",
                "Extracted Text:"
            ]
            
            full_text = "\n".join(image_info) + "\n" + extracted_text.strip()
            
            print(f"Successfully extracted text from image: {len(extracted_text)} characters")
            return full_text
            
        except Exception as e:
            error_msg = f"Error extracting text from image: {e}"
            print(error_msg)
            return f"Failed to process image file: {str(e)}. Please ensure the image is not corrupted and contains readable text."
    
    def process_url(self, url: str) -> bool:
        """Process content from a URL and store it in the database."""
        if not WEB_SCRAPING_AVAILABLE:
            print("Web scraping is not available. Please install requests, beautifulsoup4, and trafilatura.")
            return False
            
        try:
            # Validate URL
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            print(f"Fetching content from: {url}")
            
            # Use trafilatura for better content extraction
            try:
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    text = trafilatura.extract(downloaded)
                    if text:
                        # Add URL metadata
                        metadata = [
                            f"URL Content: {url}",
                            f"Extracted on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            "",
                            "Content:"
                        ]
                        full_text = "\n".join(metadata) + "\n" + text
                    else:
                        # Fallback to requests + BeautifulSoup
                        full_text = self._extract_with_beautifulsoup(url)
                else:
                    full_text = self._extract_with_beautifulsoup(url)
            except Exception:
                # Fallback to requests + BeautifulSoup
                full_text = self._extract_with_beautifulsoup(url)
            
            if not full_text or not full_text.strip():
                print("No content could be extracted from the URL")
                return False
            
            # Create a filename from URL
            filename = self._url_to_filename(url)
            
            # Split text into chunks
            chunks = self._split_text_into_chunks(full_text)
            
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
            
            print(f"Successfully processed URL {url} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            print(f"Error processing URL: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_with_beautifulsoup(self, url: str) -> str:
        """Fallback content extraction using requests and BeautifulSoup."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Add metadata
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No title"
            
            metadata = [
                f"URL Content: {url}",
                f"Title: {title_text}",
                f"Extracted on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "Content:"
            ]
            
            return "\n".join(metadata) + "\n" + text
            
        except Exception as e:
            print(f"Error with BeautifulSoup extraction: {e}")
            return ""
    
    def _url_to_filename(self, url: str) -> str:
        """Convert URL to a safe filename."""
        # Remove protocol and clean up
        filename = re.sub(r'^https?://', '', url)
        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        filename = filename[:100]  # Limit length
        return f"url_{filename}.txt"
    
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
