#test pdf processor that splitting is occuring correctly

from pdf_processor import extract_text
from pathlib import Path

def main():
    # Get path to PDF in Documents/PRDs folder
    pdf_path = str(Path.home() / "Documents" / "PRDs" / "MOCK-PRD-Arduino-Uno.pdf")
    # Extract text from the PDF
    extracted_text = extract_text(pdf_path)
    
    # Print the first 500 characters to see the result
    print("Extracted text (first 500 characters):")
    print(extracted_text[:500])
    
    # Print the total length of extracted text
    print(f"\nTotal characters extracted: {len(extracted_text)}")

if __name__ == "__main__":
    main() 