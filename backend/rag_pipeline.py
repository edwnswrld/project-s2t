# This file will contain the LangChain + FAISS RAG pipeline implementation 

from __future__ import annotations
from typing import List, Dict, Any
import json
import re
import os
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import requests
import torch

#this class is used to extract requirements from a text file
class PRDExtractor:
    def __init__(self):
        self.section_patterns = [
            r'\d+\.\d+\s*Hardware\s*Requirements',
            r'\d+\.\d+\s*Software\s*Requirements',
            r'Technical\s*Requirements'
        ]
        self.requirement_patterns = [
            r'^(?P<id>[A-Z]{2}\d+)\s+(?P<req>.+?)\s+(?P<pri>High|Medium|Low)\s+(?P<notes>.+)$',
            r'^-\s*(?P<req>.+?)\s*:\s*(?P<notes>.+)$',
            r'^(?:The system shall|Must|Should)\s+(?P<req>.+?)(?:\s*\((?P<notes>.+)\))?$'
        ]

    def extract_requirements_from_text(self, text: str) -> List[Dict[str, str]]:
        """Extract requirements from text using the defined patterns.
        
        Args:
            text: The text to extract requirements from
            
        Returns:
            List of dictionaries containing requirement information
        """
        requirements = []
        
        # Split text into lines
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try each pattern
            for pattern in self.requirement_patterns:
                match = re.match(pattern, line)
                if match:
                    req_dict = match.groupdict()
                    # Ensure all required fields are present
                    if 'id' not in req_dict:
                        req_dict['id'] = ''
                    if 'priority' not in req_dict:
                        req_dict['priority'] = 'Medium'
                    if 'notes' not in req_dict:
                        req_dict['notes'] = ''
                    requirements.append(req_dict)
                    break
        
        return requirements

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and newlines while preserving tabs."""
    # Replace multiple newlines and spaces with single ones, but preserve tabs
    text = re.sub(r'[ \n]+', ' ', text)
    # Remove markdown formatting
    text = re.sub(r'\*\*|__', '', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?\t-]', '', text)
    return text.strip()

def format_requirement_to_sentence(req: Dict[str, str]) -> str:
    """Convert structured requirement to natural language sentence.
    
    Args:
        req: Dictionary containing requirement fields (id, requirement, priority, notes)
        
    Returns:
        Formatted requirement as a natural language sentence
    """
    # First normalize whitespace
    line = ' '.join(line.split())
    
    # Handle both "HW1 Desc Priority Notes" and "ID   Requirement   Priority   Notes" formats
    parts = re.split(r'\s{2,}|\t', line)  # Split on multiple spaces or tabs
    
    if len(parts) >= 4:  # Full format
        req_id, desc, priority, notes = parts[0], ' '.join(parts[1:-2]), parts[-2], parts[-1]
    elif len(parts) == 3:  # Missing notes
        req_id, desc, priority = parts
        notes = ""
    else:
        return line  # Can't parse
    
    return f"{req_id}: Requires {desc.lower()} ({priority.lower()} priority). Notes: {notes.lower()}"
    """Convert structured requirement to natural language sentence"""
    sentence = f"{req['id'] + ': ' if req['id'] else ''}The system requires {req['requirement'].lower()}"
    
    if req['notes'] and req['notes'].lower() not in ["none", "n/a", "na"]:
        notes = re.sub(r'[^\w\s.,!?0-9-]', '', req['notes'])
        sentence += f". Additional notes: {notes.lower()}"
        
    sentence += f". This is a {req['priority'].lower()}-priority task."
    return sentence

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
        # Initialize the LLM with Ollama using callbacks instead of callback_manager
        callbacks = [StreamingStdOutCallbackHandler()]
        self.llm = Ollama(
            model="mistral",
            callbacks=callbacks,
            temperature=0.7
        )
        
        # Initialize embeddings with explicit model name and clean_up_tokenization_spaces
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'clean_up_tokenization_spaces': True}
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Increased from 1000 for fewer chunks
            chunk_overlap=100,  # Reduced from 200 for less redundancy
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]  # Added explicit separators
        )
        
        # Initialize or load vector store
        self.vector_store = None
        self.cache_dir = "vector_store_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def process_text(self, text: str) -> None:
        """Process text by splitting into chunks and creating embeddings."""
        # Clean the text first
        text = clean_text(text)
        
        # Generate a cache key based on the text content
        import hashlib
        cache_key = hashlib.md5(text.encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.faiss")
        
        # Check if we have a cached version
        if os.path.exists(cache_path):
            print("Loading cached vector store...")
            self.vector_store = FAISS.load_local(cache_path, self.embeddings, allow_dangerous_deserialization=True)
            return
            
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create FAISS vector store
        self.vector_store = FAISS.from_texts(chunks, self.embeddings)
        
        # Save to cache
        self.vector_store.save_local(cache_path)

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
        """Generate tasks from individual requirements in the PRD."""
        extractor = PRDExtractor()
        requirements = extractor.extract_requirements_from_text(context)
        
        if isinstance(requirements, str):
            print(f"Warning: {requirements}")
            return []
            
        tasks = []
        for req in requirements:
            # Format the requirement into a sentence
            formatted_req = format_requirement_to_sentence(req)
            
            # Extract technical values for this requirement
            technical_values = self.extract_technical_values(formatted_req)
            
            # Create a focused prompt for this specific requirement
            prompt = f"""<s>[INST] You are a hardware product manager. Given this specific requirement, generate a JSON task that captures its scope. The JSON should include:
- title: A concise summary of this specific requirement
- description: 1-2 sentences describing what needs to be done
- tasks: A list of 3-5 specific, actionable subtasks
- acceptance_criteria: A list of 3-5 measurable criteria for this specific requirement
- priority: Use the requirement's priority (High/Medium/Low)
- assignee: Hardware Engineer for hardware requirements, Software Engineer for software/firmware
- secondary_assignees: ["QA Engineer", "Documentation Engineer"]
- due_date: TBD

Requirement: {formatted_req}

Technical Values Found:
{json.dumps(technical_values, indent=2)}

Example of good task breakdown:
Input: "HW1: The system requires 14 digital io pins. Out of these, 6 should support pwm. This is a high-priority task."
Output:
{{
    "title": "Implement 14 Digital I/O Pins with 6 PWM Support",
    "description": "Design and implement 14 digital I/O pins, ensuring 6 of them support PWM functionality. Must meet all technical specifications.",
    "tasks": [
        "Design pin layout for all 14 digital I/O pins",
        "Configure 6 specific pins for PWM support",
        "Test PWM functionality on the configured pins",
        "Document pin specifications and PWM capabilities"
    ],
    "acceptance_criteria": [
        "All 14 digital I/O pins are functional",
        "6 pins successfully support PWM functionality",
        "PWM frequency and duty cycle meet specifications",
        "Documentation complete with pin assignments and PWM capabilities"
    ],
    "priority": "High",
    "assignee": "Hardware Engineer",
    "secondary_assignees": ["Embedded Systems Engineer", "QA Engineer"],
    "due_date": "TBD"
}}

Important Guidelines:
1. When dealing with technical specifications:
   - If a requirement mentions "X components", it means X total components, not component number X
   - If a requirement mentions "Y should support [feature]", it means Y out of the total components should have that feature
2. Always verify:
   - Total quantity of components
   - Number of components with special features
   - Any specific technical requirements or constraints
3. Ensure tasks and acceptance criteria:
   - Address all components and their quantities
   - Verify all special features and capabilities
   - Include proper testing and documentation requirements

Now, convert the requirement into a similar task structure. Focus on one step at a time:
1. First, identify the key components and their quantities
2. Then, extract technical values and their relationships
3. Next, create specific tasks that address all requirements
4. Finally, define acceptance criteria that verify all specifications

Output only valid JSON that can be parsed by json.loads(). Do not include any other text or comments. [/INST]</s>"""

            # Generate task using Ollama
            generated_text = self.generate_with_api(prompt)
            
            # Extract JSON from the generated text
            try:
                json_match = re.search(r'\{[\s\S]*\}', generated_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    task_json = json.loads(json_str)
                    tasks.append(task_json)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON for requirement {req.get('id', 'unknown')}: {str(e)}")
            except Exception as e:
                print(f"Error processing requirement {req.get('id', 'unknown')}: {str(e)}")
        
        return tasks

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