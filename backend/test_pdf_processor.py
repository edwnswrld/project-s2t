#test pdf processor that splitting is occuring correctly

from pdf_processor import extract_text

def main():
    # Replace 'path_to_your_pdf.pdf' with the actual path to your PDF file
    pdf_path = "/Users/edwinguzman/Downloads/test1_SPECIFICATION_DOCUMENT.pdf"  # You'll need to put your PDF in this location
    
    # Extract text from the PDF
    extracted_text = extract_text(pdf_path)
    
    # Print the first 500 characters to see the result
    print("Extracted text (first 500 characters):")
    print(extracted_text[:500])
    
    # Print the total length of extracted text
    print(f"\nTotal characters extracted: {len(extracted_text)}")

if __name__ == "__main__":
    main() 