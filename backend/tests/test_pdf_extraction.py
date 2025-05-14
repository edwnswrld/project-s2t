#test that looks for requirements in pdf
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from pdf_processor import PRDExtractor

def print_requirements(requirements, section_name):
    print(f"\nExtracted {section_name.title()} Requirements:")
    if isinstance(requirements, list):
        if requirements:
            print("ID Requirement Priority Notes")
            for req in requirements:
                print(f"{req['id']} {req['requirement']} {req['priority']} {req['notes']}")
        else:
            print(f"No {section_name} requirements found.")
    else:
        print(requirements)

if __name__ == "__main__":
    pdf_path = str(Path.home() / "Documents" / "PRDs" / "MOCK-PRD-Arduino-Uno.pdf")
    extractor = PRDExtractor()
    results = extractor.extract_requirements(pdf_path)
    
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print_requirements(results["hardware"], "hardware")
        print_requirements(results["software"], "software")
        print_requirements(results["firmware"], "firmware") 