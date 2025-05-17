import os
from pathlib import Path
import PyPDF2
import re
import logging
from typing import Optional
from functools import wraps
import time
from backend.rag_pipeline import RAGPipeline, clean_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry operations on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

def validate_pdf_file(pdf_path: str) -> Optional[str]:
    """Validate PDF file before processing.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Error message if validation fails, None if successful
    """
    if not os.path.exists(pdf_path):
        return f"PDF file not found at {pdf_path}"
        
    if not pdf_path.lower().endswith('.pdf'):
        return "File must be a PDF document"
        
    file_size = os.path.getsize(pdf_path)
    if file_size > 10 * 1024 * 1024:  # 10MB limit
        return f"File size ({file_size/1024/1024:.2f}MB) exceeds 10MB limit"
        
    if file_size == 0:
        return "PDF file is empty"
        
    return None

@retry_on_failure(max_retries=3)
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file with improved error handling.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text from the PDF
        
    Raises:
        ValueError: If PDF validation fails
        PyPDF2.PdfReadError: If PDF is corrupted or encrypted
        IOError: If file cannot be read
    """
    # Validate PDF file
    if error_msg := validate_pdf_file(pdf_path):
        raise ValueError(error_msg)
        
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Verify PDF isn't encrypted
            if pdf_reader.is_encrypted:
                raise PyPDF2.PdfReadError("Encrypted PDFs are not supported")
                
            # Verify PDF has pages
            if len(pdf_reader.pages) == 0:
                raise PyPDF2.PdfReadError("PDF contains no pages")
            
            # Extract text from each page
            for i, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if not page_text:
                        logger.warning(f"No text extracted from page {i+1}")
                        continue
                        
                    logger.info(f"Successfully extracted text from page {i+1}")
                    
                    # First, preserve requirement IDs by adding spaces around them
                    page_text = re.sub(r'(HW\d+|FW\d+)', r' \1 ', page_text)
                    
                    # Clean up the text to preserve tabular structure
                    page_text = re.sub(r'[ ]+', ' ', page_text)  # Only replace spaces, not tabs
                    
                    # Ensure consistent line endings
                    page_text = page_text.replace('\r\n', '\n').replace('\r', '\n')
                    
                    # Add explicit tab between requirement ID and description
                    page_text = re.sub(r'(HW\d+|FW\d+)\s+', r'\1\t', page_text)
                    
                    # Add double newline before section headers
                    page_text = re.sub(r'(\d+\.\d+\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)*\s*Requirements)', r'\n\n\1', page_text)
                    
                    text += page_text + "\n"
                    
                except Exception as e:
                    logger.error(f"Error processing page {i+1}: {str(e)}")
                    continue
                
    except PyPDF2.PdfReadError as e:
        logger.error(f"PDF reading error: {str(e)}")
        raise
    except IOError as e:
        logger.error(f"File I/O error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise
        
    if not text.strip():
        raise ValueError("No text could be extracted from the PDF")
        
    return text

def process_prd(pdf_path: str) -> None:
    """Process a PRD PDF file and generate tasks.
    
    Args:
        pdf_path: Path to the PDF file
    """
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Extract text from PDF
    print(f"Extracting text from {pdf_path}...")
    text = extract_text_from_pdf(pdf_path)
    
    if not text or text == "Can not read PDF":
        print("No text extracted from PDF. Exiting...")
        return
        
    # Clean the extracted text
    cleaned_text = clean_text(text)
    print("\nCleaned text:")
    print("=" * 80)
    print(cleaned_text)
    print("=" * 80)
    
    # Process the text through RAG pipeline
    print("\nProcessing text through RAG pipeline...")
    rag.process_text(cleaned_text)
    
    # Generate tasks
    print("\nGenerating tasks...")
    tasks = rag.generate_tasks(cleaned_text)
    
    if not tasks:
        print("\nNo tasks were generated. This could be because:")
        print("1. The PDF doesn't contain requirements in the expected format")
        print("2. The requirements section is not properly formatted")
        print("3. The text extraction didn't preserve the formatting")
        return
    
    # Print tasks
    print("\nGenerated Tasks:")
    print("=" * 80)
    for i, task in enumerate(tasks, 1):
        print(f"\nTask {i}:")
        print("-" * 40)
        print(f"Title: {task['title']}")
        print(f"Description: {task['description']}")
        print("\nSubtasks:")
        for subtask in task['tasks']:
            print(f"- {subtask}")
        print("\nAcceptance Criteria:")
        for criterion in task['acceptance_criteria']:
            print(f"- {criterion}")
        print(f"\nPriority: {task['priority']}")
        print(f"Assignee: {task['assignee']}")
        if 'secondary_assignees' in task:
            print(f"Secondary Assignees: {', '.join(task['secondary_assignees'])}")
        print(f"Due Date: {task['due_date']}")
        print("-" * 40)

def main():
    # Get the user's documents folder path
    documents_path = str(Path.home() / "Documents" / "PRDs")
    pdf_path = os.path.join(documents_path, "MOCK-PRD-Arduino-Uno.pdf")
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"Error: Could not find {pdf_path}")
        return
        
    # Process the PRD
    process_prd(pdf_path)

if __name__ == "__main__":
    main() 