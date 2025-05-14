import unittest
from backend.rag_pipeline import RAGPipeline, clean_text

class TestRequirementTasks(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.rag = RAGPipeline()

    def test_hardware_requirement(self):
        """Test processing a hardware requirement"""
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
        
        # Verify task structure
        task = tasks[0]
        self.assertIn('title', task, "Task should have a title")
        self.assertIn('description', task, "Task should have a description")
        self.assertIn('tasks', task, "Task should have subtasks")
        self.assertIn('acceptance_criteria', task, "Task should have acceptance criteria")
        self.assertIn('priority', task, "Task should have a priority")
        self.assertIn('assignee', task, "Task should have an assignee")
        
        # Verify task content
        self.assertTrue(len(task['tasks']) >= 3, "Task should have at least 3 subtasks")
        self.assertTrue(len(task['acceptance_criteria']) >= 3, "Task should have at least 3 acceptance criteria")
        self.assertEqual(task['priority'], "High", "Task priority should match requirement")
        self.assertEqual(task['assignee'], "Hardware Engineer", "Task should be assigned to Hardware Engineer")
        
        # Print the generated task
        print("\nGenerated Task:")
        print("=" * 40)
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
        print("-" * 40)

if __name__ == '__main__':
    unittest.main() 