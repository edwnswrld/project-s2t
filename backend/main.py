from pdf_processor import extract_text

def main():
    text = extract_text("data/specs/sample_spec.pdf")
    print("Setup complete. Extracted text:", text[:100])

if __name__ == "__main__":
    main() 