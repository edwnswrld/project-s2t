#returns a numbered list of the sections that ragpipe found requirements
import unittest
from pathlib import Path
import os
from backend.process_prd import process_prd, extract_text_from_pdf
from backend.rag_pipeline import RAGPipeline, clean_text

class TestPRDProcessing(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.documents_path = str(Path.home() / "Documents" / "PRDs")
        self.test_pdf_path = os.path.join(self.documents_path, "MOCK-PRD-Arduino-Uno.pdf")
        self.rag = RAGPipeline()

    def test_requirement_sections_found(self):
        """Test that requirement sections are found in the PRD"""
        # Extract text from PDF
        text = extract_text_from_pdf(self.test_pdf_path)
        self.assertNotEqual(text, "Can not read PDF", "PDF should be readable")
        self.assertTrue(len(text) > 0, "PDF should contain text")

        # Clean the text
        cleaned_text = clean_text(text)
        
        # Process through RAG pipeline
        self.rag.process_text(cleaned_text)
        
        # Generate tasks
        tasks = self.rag.generate_tasks(cleaned_text)
        
        # Verify that tasks were generated
        self.assertIsNotNone(tasks, "Tasks should be generated")
        self.assertTrue(len(tasks) > 0, "At least one task should be generated")
        
        # Print numbered list of requirement sections found
        print("\nRequirement Sections Found:")
        print("=" * 40)
        for i, task in enumerate(tasks, 1):
            print(f"{i}. {task['title']}")
            print(f"   Description: {task['description'][:100]}...")  # Print first 100 chars
            print(f"   Number of subtasks: {len(task['tasks'])}")
            print(f"   Number of acceptance criteria: {len(task['acceptance_criteria'])}")
            print("-" * 40)

if __name__ == '__main__':
    unittest.main() 

#Main Test Method:
# test_requirement_sections_found is our main test
# It follows these steps:
# - Extracts text from the PDF
# - Cleans the text 
# - Processes it through the RAG pipeline
# - Generates tasks
# - Verifies that tasks were found
# - Prints a nice numbered list of all requirement sections found

# Output Format:
# For each requirement section found, it will show:
# - A number
# - The title
# - A preview of the description
# - Number of subtasks
# - Number of acceptance criteria
