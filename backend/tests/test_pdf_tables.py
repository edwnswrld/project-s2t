import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from pdf_processor import extract_text

def test_extract_tables():
    """
    Test function to extract and display tables from the PDF.
    """
    # Path to the PDF file
    pdf_path = str(Path.home() / "Documents" / "PRDs" / "MOCK-PRD-Arduino-Uno.pdf")
    
    # Extract text from the PDF
    extracted_text = extract_text(pdf_path)
    
    # Print the extracted text
    print("\nExtracted Text from PDF Tables:")
    print("-" * 50)
    print(extracted_text if extracted_text else "No tables found in the PDF.")
    print("-" * 50)

if __name__ == "__main__":
    test_extract_tables() 