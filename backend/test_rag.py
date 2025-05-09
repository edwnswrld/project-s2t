# hardcoded test requirements to test if llm can generate json formatted tasks

from pdf_processor import extract_text
from rag_pipeline import RAGPipeline
import json

# Test the task generation from a requirement
def test_task_generation():
    rag = RAGPipeline()
    
    # Test requirement with structured format
    requirement = """Features & Requirements:
requirement ID	Requirement	Priority	Notes
HW1	14 Digital I/O Pins	High	6 should support PWM"""
    
    print("\nInput Requirement:")
    print(requirement)
    print("\n" + "="*80)
    
    # Generate tasks
    tasks = rag.generate_tasks(requirement)
    
    print("\nGenerated Tasks (JSON format):")
    print(json.dumps(tasks, indent=2))
    print("\n" + "="*80)

if __name__ == "__main__":
    test_task_generation()

