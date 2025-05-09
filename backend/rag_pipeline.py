# This file will contain the LangChain + FAISS RAG pipeline implementation 

from __future__ import annotations
from typing import List, Dict, Any
import json
import re
import os
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import requests
import torch

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and newlines."""
    # Replace multiple newlines and spaces with single ones
    text = re.sub(r'\s+', ' ', text)
    # Remove markdown formatting
    text = re.sub(r'\*\*|__', '', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def is_likely_task(text: str) -> bool:
    """Determine if a piece of text is likely to be a task."""
    # Skip if too short or too long
    words = text.split()
    if len(words) < 3 or len(words) > 50:
        return False
        
    # Skip if starts with common non-task patterns
    non_task_starts = [
        'appendix', 'chapter', 'section', 'table of', 'contents',
        'introduction', 'overview', 'summary', 'definition',
        'appendices', 'revision history', 'references', 'glossary',
        'index', 'figure', 'table', 'note:', 'warning:', 'caution:',
        'product description', 'market needs', 'key pain points',
        'physical design', 'bootloader', 'assembly', 'testing', 
        'document', 'prd', 'version', 'improvements', 'definitions', 'abbreviations',
        'release plan', 'customer satisfaction', 'supported libraries',
        'variant', 'support', 'resolution', 'speed', 'memory',
        'dimensions', 'power', 'consumption', 'features'
    ]
    if any(text.lower().startswith(word) for word in non_task_starts):
        return False
        
    # Skip if it's just a number or section reference
    if re.match(r'^\d+(\.\d+)*\s*$', text):
        return False
        
    # Skip if it's just a heading pattern
    if re.match(r'^\d+\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)*\s*$', text):
        return False
        
    # Skip if it's just a statement (no action)
    if text.lower().startswith(('it ', 'this ', 'the ', 'these ', 'in ', 'at ', 'on ', 'with ')):
        return False
        
    # Skip if it's just a measurement or technical spec
    if re.match(r'^[\d\.]+(mA|V|MHz|KB|MB|GB|mm|cm|in)\b', text):
        return False
        
    # Must contain an action word or be a user story/requirement
    action_words = [
        'implement', 'create', 'develop', 'test', 'verify', 'ensure',
        'must', 'should', 'needs', 'requires', 'design', 'build',
        'integrate', 'deploy', 'update', 'modify', 'add', 'remove',
        'configure', 'setup', 'install', 'validate', 'review',
        'improve', 'optimize', 'fix', 'debug', 'document',
        'support', 'enable', 'provide', 'maintain', 'achieve',
        'complete', 'deliver', 'finalize', 'prepare', 'establish'
    ]
    has_action = any(word in text.lower() for word in action_words)
    
    # Check for user story format
    is_user_story = text.lower().startswith('as a') and ('i want' in text.lower() or 'i need' in text.lower())
    
    # Check for requirement format
    requirement_starts = ['shall', 'must', 'will', 'should', 'needs to', 'required to']
    is_requirement = any(text.lower().startswith(word) for word in requirement_starts)
    
    # Check for bullet points with requirements
    is_bullet_requirement = (
        text.strip().startswith('-') and 
        any(word in text.lower() for word in action_words + ['need', 'want'] + requirement_starts) and
        not any(text.lower().endswith(word) for word in ['mode', 'support', 'reviews', 'variant'])
    )
    
    return has_action or is_user_story or is_requirement or is_bullet_requirement

def extract_date(text: str) -> str:
    """Extract date information from text if present."""
    # Look for common date patterns
    date_patterns = [
        r'by (\w+ \d{1,2}(?:st|nd|rd|th)?, \d{4})',
        r'due (\w+ \d{1,2}(?:st|nd|rd|th)?, \d{4})',
        r'before (\w+ \d{1,2}(?:st|nd|rd|th)?, \d{4})',
        r'(\d{1,2}/\d{1,2}/\d{2,4})',
        r'(\d{4}-\d{2}-\d{2})'
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # Look for relative time mentions
    time_patterns = {
        r'next month': 'TBD (Next Month)',
        r'next week': 'TBD (Next Week)',
        r'(\d+) months?': lambda m: f"TBD ({m.group(1)} months)",
        r'(\d+) weeks?': lambda m: f"TBD ({m.group(1)} weeks)"
    }
    
    for pattern, handler in time_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return handler(match) if callable(handler) else handler
            
    return "TBD"

class RAGPipeline:
    def __init__(self):
        # Initialize the LLM with Ollama
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.llm = Ollama(
            model="mistral",
            callback_manager=callback_manager,
            temperature=0.7
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize or load vector store
        self.vector_store = None

    def process_text(self, text: str) -> None:
        """Process text by splitting into chunks and creating embeddings."""
        # Clean the text first
        text = clean_text(text)
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create FAISS vector store
        self.vector_store = FAISS.from_texts(chunks, self.embeddings)

    def extract_tasks_from_text(self, text: str) -> List[Dict[str, str]]:
        """Extract tasks from text using a structured approach."""
        # Clean the text first
        text = clean_text(text)
        # Split text into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        tasks = []
        
        for sentence in sentences:
            # Skip if not likely to be a task
            if not is_likely_task(sentence):
                continue
                
            # Determine priority based on keywords
            priority = "Medium"
            if any(word in sentence.lower() for word in ['urgent', 'critical', 'immediate', 'asap', 'high priority', 'important', 'crucial']):
                priority = "High"
            elif any(word in sentence.lower() for word in ['later', 'eventually', 'could', 'might', 'low priority', 'optional', 'nice to have']):
                priority = "Low"
                
            # Extract assignee if mentioned
            assignee = "Development Team"
            teams = [
                'circuit team', 'development team', 'testing team',
                'technical writing team', 'hardware team', 'software team',
                'qa team', 'design team', 'documentation team'
            ]
            for team in teams:
                if team in sentence.lower():
                    assignee = team.title()
                    break
            
            # Extract any dates mentioned
            due_date = extract_date(sentence)
            
            # Create task with cleaned text
            task = {
                "title": clean_text(sentence[:50]) + ('...' if len(sentence) > 50 else ''),
                "description": clean_text(sentence),
                "tasks": [
                    f"Analyze requirements from: {clean_text(sentence)}",
                    "Break down into actionable steps",
                    "Create implementation plan",
                    "Review with stakeholders"
                ],
                "acceptance criteria": [
                    "All requirements from the description are met",
                    "Implementation follows best practices",
                    "Documentation is complete and clear",
                    "Testing confirms functionality"
                ],
                "priority": priority,
                "assignee": assignee,
                "due_date": due_date
            }
            tasks.append(task)
        
        return tasks

    def extract_technical_values(self, text: str) -> dict:
        """Extract numerical specifications and technical values from text."""
        values = {}
        
        # Extract numbers with units
        patterns = {
            'power': r'(\d+)\s*(?:hp|HP|horsepower|watt|W)',
            'voltage': r'(\d+)\s*(?:V|v|volt|volts)',
            'current': r'(\d+)\s*(?:A|a|amp|amps|ampere|amperes)',
            'frequency': r'(\d+)\s*(?:Hz|hz|hertz|MHz|Mhz|GHz|Ghz)',
            'pins': r'(\d+)\s*(?:pins?|I/O|digital|analog)',
            'memory': r'(\d+)\s*(?:KB|MB|GB|kb|mb|gb)',
            'dimension': r'(\d+)\s*(?:mm|cm|m|inch|in)',
            'temperature': r'(\d+)\s*(?:°C|°F|C|F|celsius|fahrenheit)',
            'generic_number': r'(\d+)(?:\s+[a-zA-Z]+)?'  # Fallback for any number
        }
        
        for key, pattern in patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if key not in values:
                    values[key] = []
                values[key].append(match.group(0))
        
        return values

    def generate_with_api(self, prompt: str) -> str:
        """Generate text using Ollama."""
        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            print(f"Ollama Error: {str(e)}")
            return ""

    def generate_tasks(self, context: str) -> List[Dict[str, str]]:
        """Generate tasks from the provided context using a hybrid approach."""
        # Clean the input context
        context = clean_text(context)
        
        # Step 1: Extract and validate requirements section
        section_patterns = [
            r'Features\s*&\s*Requirements.*?(?=\n\n|\Z)',
            r'Features\s*and\s*Requirements.*?(?=\n\n|\Z)',
            r'Requirements.*?(?=\n\n|\Z)',
            r'Technical\s*Requirements.*?(?=\n\n|\Z)'
        ]
        
        requirements_text = ""
        for pattern in section_patterns:
            match = re.search(pattern, context, re.IGNORECASE | re.DOTALL)
            if match:
                requirements_text = match.group(0)
                break
        
        if not requirements_text:
            print("Warning: No Features & Requirements section found in the text")
            return []

        # Step 2: Extract technical values
        technical_values = self.extract_technical_values(requirements_text)
        
        # Step 3: Use FAISS to find similar requirements and their implementations
        if self.vector_store:
            similar_chunks = self.search_similar_chunks(requirements_text, k=2)
            similar_context = "\nSimilar Requirements Found:\n" + "\n".join(similar_chunks)
        else:
            similar_context = ""

        # Create a more focused prompt for Mistral
        prompt = f"""<s>[INST] You are a technical product manager. Convert the following requirement into a structured task:

Input Requirement:
{requirements_text}

Technical Values Found:
{json.dumps(technical_values, indent=2)}
{similar_context}

Example of good task breakdown:
Input: "14 Digital I/O Pins, 6 with PWM support"
Output:
{{
    "title": "Implement 14 Digital I/O Pins with 6 PWM Support",
    "description": "Design and implement 14 digital I/O pins, ensuring 6 support PWM functionality. Must meet all technical specifications.",
    "tasks": [
        "Design pin layout for 14 digital I/O pins",
        "Configure 6 pins for PWM support",
        "Test PWM functionality",
        "Document specifications"
    ],
    "acceptance_criteria": [
        "All 14 pins functional",
        "6 pins support PWM",
        "PWM frequency meets specs",
        "Documentation complete"
    ],
    "priority": "High",
    "assignee": "Hardware Engineer",
    "secondary_assignees": ["Embedded Systems Engineer", "QA Engineer"],
    "due_date": "TBD"
}}

Now, convert the requirement into a similar task structure. Focus on one step at a time:
1. First, identify the key components
2. Then, extract technical values
3. Next, create specific tasks
4. Finally, define acceptance criteria

Output only valid JSON that can be parsed by json.loads() [/INST]</s>"""

        # Generate tasks using Ollama
        generated_text = self.generate_with_api(prompt)
        
        print("\nDebug - Mistral Generated Output:")
        print("="*80)
        print(generated_text)
        print("="*80)
        
        # Extract JSON from the generated text
        try:
            # Find JSON-like structure in the text
            json_match = re.search(r'\{[\s\S]*\}', generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                print("\nDebug - Extracted JSON string:")
                print(json_str)
                print("="*80)
                
                # Try to parse the JSON
                task_json = json.loads(json_str)
                return [task_json]
        except json.JSONDecodeError as e:
            print(f"\nDebug - JSON Decode Error: {str(e)}")
        except Exception as e:
            print(f"\nDebug - Unexpected error: {str(e)}")
        
        # If JSON parsing fails, create a structured task from the text
        requirement_lines = requirements_text.split('\n')
        if len(requirement_lines) >= 2:
            requirement_content = requirement_lines[1]  # Get the actual requirement line
            fields = requirement_content.split('\t')
            if len(fields) >= 4:
                req_id, req_text, priority, notes = fields
                
                # Create a detailed task based on the requirement
                task = {
                    'title': f"Define and Validate {req_text}",
                    'description': f"Design and validate the configuration to provide {req_text}, with {notes}",
                    'tasks': [
                        f"Select and configure components for {req_text}",
                        f"Implement {notes} functionality",
                        f"Verify {notes} through testing and validation",
                        "Create technical documentation with specifications"
                    ],
                    'acceptance_criteria': [
                        f"All {req_text} are functional and meet specifications",
                        f"{notes} functionality verified and tested",
                        "Passes all technical validation tests",
                        "Documentation complete with all specifications"
                    ],
                    'priority': priority,
                    'assignee': 'Hardware Engineer (Lead)',
                    'secondary_assignees': ['Embedded Systems Engineer', 'QA Engineer'],
                    'due_date': 'TBD'
                }
                return [task]
        
        return []

    def search_similar_chunks(self, query: str, k: int = 3) -> List[str]:
        """Search for similar chunks in the vector store."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call process_text first.")
        
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

def main():
    # Example usage
    rag = RAGPipeline()
    
    # Example text
    sample_text = """
    Project Specification:
    The system requires urgent implementation of a power supply test module.
    The circuit team needs to verify voltage stability by next month.
    Documentation should be completed by the technical writing team.
    Security audit is critical and must be performed before deployment.
    """
    
    # Process the text
    rag.process_text(sample_text)
    
    # Generate tasks
    tasks = rag.generate_tasks(sample_text)
    
    # Print tasks as JSON
    print(json.dumps(tasks, indent=2))

if __name__ == "__main__":
    main() 