# Test single requirement processing
import unittest
from backend.rag_pipeline import RAGPipeline, clean_text

class TestSingleRequirement(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.rag = RAGPipeline()

    def test_single_requirement(self):
        """Test processing a single requirement"""
        # Test requirement with structured format
        requirement = """Hardware Requirements:
HW1\t14 Digital I/O Pins\tHigh\t6 should support PWM"""
        
        # Clean the text
        cleaned_text = clean_text(requirement)
        
        # Process through RAG pipeline
        self.rag.process_text(cleaned_text)
        
        # Generate tasks
        tasks = self.rag.generate_tasks(cleaned_text)
        
        # Verify that tasks were generated
        self.assertIsNotNone(tasks, "Tasks should be generated")
        self.assertTrue(len(tasks) > 0, "At least one task should be generated")
        
        # Print the generated task
        print("\nGenerated Task:")
        print("=" * 40)
        for task in tasks:
            print(f"Title: {task['title']}")
            print(f"Description: {task['description']}")
            print(f"Priority: {task['priority']}")
            print("-" * 40)

if __name__ == '__main__':
    unittest.main()

