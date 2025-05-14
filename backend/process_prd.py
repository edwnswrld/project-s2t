import os
from pathlib import Path
import PyPDF2
import re
from backend.rag_pipeline import RAGPipeline, clean_text


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text from the PDF
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from each page
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                print(f"\nPage {i+1} content:")
                print("-" * 40)
                print(page_text)
                print("-" * 40)
                
                # First, preserve requirement IDs by adding spaces around them
                page_text = re.sub(r'(HW\d+|FW\d+)', r' \1 ', page_text)
                
                # Clean up the text to preserve tabular structure
                # Replace multiple spaces with a single space, but preserve tabs
                page_text = re.sub(r'[ ]+', ' ', page_text)  # Only replace spaces, not tabs
                
                # Ensure consistent line endings
                page_text = page_text.replace('\r\n', '\n').replace('\r', '\n')
                
                # Add explicit tab between requirement ID and description
                page_text = re.sub(r'(HW\d+|FW\d+)\s+', r'\1\t', page_text)
                
                # Add double newline before section headers
                page_text = re.sub(r'(\d+\.\d+\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)*\s*Requirements)', r'\n\n\1', page_text)
                
                text += page_text + "\n"
                
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return "Can not read PDF"
        
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