"""
Test file for validating task generation against perfect output examples.
"""

import unittest
import asyncio
from backend.examples.perfect_outputs import get_example_outputs
from backend.rag_pipeline import RAGPipeline

class TestPerfectOutputs(unittest.TestCase):
    def setUp(self):
        self.rag = RAGPipeline()
        self.examples = get_example_outputs()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_hardware_requirement(self):
        """Test hardware requirement processing against perfect output."""
        example = self.examples["hardware"]
        tasks = self.loop.run_until_complete(self.rag.generate_tasks(example["input"]))
        
        # Compare generated task with expected output
        self.assertIsNotNone(tasks)
        self.assertTrue(len(tasks) > 0)
        
        task = tasks[0]
        expected = example["expected_output"]
        
        # Compare key fields
        self.assertEqual(task["title"], expected["title"])
        self.assertEqual(task["priority"], expected["priority"])
        self.assertEqual(task["assignee"], expected["assignee"])
        
        # Compare tasks and acceptance criteria
        self.assertEqual(len(task["tasks"]), len(expected["tasks"]))
        self.assertEqual(len(task["acceptance_criteria"]), len(expected["acceptance_criteria"]))

    def test_software_requirement(self):
        """Test software requirement processing against perfect output."""
        example = self.examples["software"]
        tasks = self.loop.run_until_complete(self.rag.generate_tasks(example["input"]))
        
        self.assertIsNotNone(tasks)
        self.assertTrue(len(tasks) > 0)
        
        task = tasks[0]
        expected = example["expected_output"]
        
        # Compare key fields
        self.assertEqual(task["title"], expected["title"])
        self.assertEqual(task["priority"], expected["priority"])
        self.assertEqual(task["assignee"], expected["assignee"])

    def test_table_requirement(self):
        """Test table format requirement processing against perfect output."""
        example = self.examples["table"]
        tasks = self.loop.run_until_complete(self.rag.generate_tasks(example["input"]))
        
        self.assertIsNotNone(tasks)
        self.assertTrue(len(tasks) > 0)
        
        task = tasks[0]
        expected = example["expected_output"]
        
        # Compare key fields
        self.assertEqual(task["title"], expected["title"])
        self.assertEqual(task["priority"], expected["priority"])
        self.assertEqual(task["assignee"], expected["assignee"])

    def test_complex_requirement(self):
        """Test complex requirement processing against perfect output."""
        example = self.examples["complex"]
        tasks = self.loop.run_until_complete(self.rag.generate_tasks(example["input"]))
        
        self.assertIsNotNone(tasks)
        self.assertTrue(len(tasks) > 0)
        
        task = tasks[0]
        expected = example["expected_output"]
        
        # Compare key fields
        self.assertEqual(task["title"], expected["title"])
        self.assertEqual(task["priority"], expected["priority"])
        self.assertEqual(task["assignee"], expected["assignee"])

    def test_documentation_requirement(self):
        """Test documentation requirement processing against perfect output."""
        example = self.examples["documentation"]
        tasks = self.loop.run_until_complete(self.rag.generate_tasks(example["input"]))
        
        self.assertIsNotNone(tasks)
        self.assertTrue(len(tasks) > 0)
        
        task = tasks[0]
        expected = example["expected_output"]
        
        # Compare key fields
        self.assertEqual(task["title"], expected["title"])
        self.assertEqual(task["priority"], expected["priority"])
        self.assertEqual(task["assignee"], expected["assignee"])

if __name__ == '__main__':
    unittest.main() 