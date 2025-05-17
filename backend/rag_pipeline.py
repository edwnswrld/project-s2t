# This file will contain the LangChain + FAISS RAG pipeline implementation 

from __future__ import annotations
from typing import List, Dict, Any
import json
import re
import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import requests
import torch
from backend.extractors.prd_extractor import PRDExtractor
from backend.examples.perfect_outputs import get_example_outputs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and newlines while preserving tabs."""
    # Replace multiple newlines and spaces with single ones, but preserve tabs
    text = re.sub(r'[ \n]+', ' ', text)
    # Remove markdown formatting
    text = re.sub(r'\*\*|__', '', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?\t-]', '', text)
    return text.strip()

def format_requirement_to_sentence(requirement: Dict[str, str]) -> str:
    """Convert structured requirement to natural language sentence.
    
    Args:
        requirement: Dictionary containing requirement fields (id, requirement, priority, notes)
            
    Returns:
        Formatted requirement as a natural language sentence
    """
    # Validate required dictionary keys
    if not all(k in requirement for k in ["requirement", "priority"]):
        print("Skipping incomplete requirement:", requirement)
        return ""
    
    # If input is a string, try to parse it
    if isinstance(requirement, str):
        # First normalize whitespace
        raw_text = ' '.join(requirement.split())
        
        # Try to parse structured format (HW1\tDesc\tPriority\tNotes)
        parts = re.split(r'\s{2,}|\t', raw_text)
        
        if len(parts) >= 4:  # Full format with tabs/spaces
            req_id, desc, priority, notes = parts[0], ' '.join(parts[1:-2]), parts[-2], parts[-1]
            return f"{req_id}: Requires {desc.lower()} ({priority.lower()} priority). Notes: {notes.lower()}"
        elif len(parts) == 3:  # Missing notes
            req_id, desc, priority = parts
            return f"{req_id}: Requires {desc.lower()} ({priority.lower()} priority)."
        else:
            # If not structured format, return cleaned string
            return clean_text(raw_text)
    
    # If input is a dictionary, use the structured approach
    sentence = f"{requirement['id'] + ': ' if requirement.get('id') else ''}The system requires {requirement['requirement'].lower()}"
    
    if requirement.get('notes') and requirement['notes'].lower() not in ["none", "n/a", "na"]:
        notes = re.sub(r'[^\w\s.,!?0-9-]', '', requirement['notes'])
        sentence += f". Additional notes: {notes.lower()}"
        
    sentence += f". This is a {requirement['priority'].lower()}-priority task."
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
        
        # Initialize text splitter with optimized parameters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Initialize vector store with improved caching
        self.vector_store = None
        self.cache_dir = "vector_store_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)

    @lru_cache(maxsize=100)
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for the text content."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()

    def process_text(self, text: str) -> None:
        """Process text by splitting into chunks and creating embeddings with improved caching."""
        # Clean the text first
        text = clean_text(text)
        
        # Generate cache key
        cache_key = self._get_cache_key(text)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.faiss")
        
        # Check if we have a cached version
        if os.path.exists(cache_path):
            logger.info("Loading cached vector store...")
            self.vector_store = FAISS.load_local(cache_path, self.embeddings, allow_dangerous_deserialization=True)
            return
            
        # Split text into chunks in parallel
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

    async def generate_with_api_async(self, prompt: str) -> str:
        """Generate text using Ollama asynchronously."""
        try:
            # Run the LLM call in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: self.llm.invoke(prompt)
            )
            return response
        except Exception as e:
            logger.error(f"Ollama Error: {str(e)}")
            return ""

    async def _process_single_requirement(self, requirement: Dict[str, str]) -> Dict[str, str]:
        """Process a single requirement asynchronously."""
        try:
            formatted_requirement = format_requirement_to_sentence(requirement)
            if not formatted_requirement:
                return None
                
            technical_values = self.extract_technical_values(formatted_requirement)
            prompt = self._build_task_prompt(formatted_requirement, technical_values, requirement)
            generated_text = await self.generate_with_api_async(prompt)
            
            return self._parse_generated_task(generated_text)
        except Exception as e:
            logger.error(f"Error processing requirement: {str(e)}")
            return None

    def _build_task_prompt(self, formatted_requirement: str, technical_values: dict, requirement: Dict[str, str]) -> str:
        """Build the prompt for task generation."""
        # Get perfect examples
        examples = get_example_outputs()
        
        # Determine requirement type
        is_hardware = "HW" in requirement.get('id', '')
        example_type = "hardware" if is_hardware else "software"
        example = examples[example_type]
        
        return f"""You are a task generator. Convert this requirement into a structured task.

Input Requirement: {formatted_requirement}

Technical Values Found:
{json.dumps(technical_values, indent=2)}

Here's an example of a perfect output for this type of requirement:
Input: {example['input']}
Output: {json.dumps(example['expected_output'], indent=2)}

Generate a task with this exact JSON structure:
{{
    "title": "A clear, concise title for the task",
    "description": "A detailed description of what needs to be done",
    "tasks": [
        "Specific subtask 1",
        "Specific subtask 2",
        "Specific subtask 3"
    ],
    "acceptance_criteria": [
        "Measurable criterion 1",
        "Measurable criterion 2",
        "Measurable criterion 3"
    ],
    "priority": "{requirement.get('priority', 'Medium')}",
    "assignee": "Hardware Engineer" if "HW" in requirement.get('id', '') else "Software Engineer",
    "due_date": "TBD"
}}

Follow the example's style and level of detail for your response."""

    def _parse_generated_task(self, generated_text: str) -> Dict[str, str]:
        """Parse the generated task from the LLM response."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
        return None

    async def generate_tasks(self, context: str) -> List[Dict[str, str]]:
        """Generate tasks from individual requirements in the PRD using async processing."""
        extractor = PRDExtractor()
        requirements = extractor.extract_requirements_from_text(context)
        
        if isinstance(requirements, str):
            logger.warning(f"Warning: {requirements}")
            return []
            
        # Process requirements concurrently
        tasks = await asyncio.gather(
            *[self._process_single_requirement(req) for req in requirements]
        )
        
        # Filter out None results
        return [task for task in tasks if task is not None]

    def search_similar_chunks(self, query: str, k: int = 3) -> List[str]:
        """Search for similar chunks in the vector store."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call process_text first.")
        
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def __del__(self):
        """Cleanup thread pool on object destruction."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

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