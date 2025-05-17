import pdfplumber
import re
from typing import List, Dict, Optional, Union
from rapidfuzz import fuzz, process
from concurrent.futures import ThreadPoolExecutor
import json

class PRDExtractor:
    def __init__(self):
        # Initialize patterns before other methods
        self.section_patterns = []
        self.requirement_patterns = []
        
        # Common requirement keywords for filtering
        self.requirement_keywords = [
            'should', 'must', 'shall', 'will', 'needs to', 'required to',
            'support', 'require', 'implement', 'provide', 'ensure',
            'verify', 'validate', 'test', 'check', 'confirm'
        ]
        
        # Common requirement formats for fuzzy matching
        self.requirement_formats = [
            "The system shall {requirement}",
            "Must {requirement}",
            "Should {requirement}",
            "Requires {requirement}",
            "Needs to {requirement}",
            "Will {requirement}"
        ]

        # Candidate section headers with their variations
        self.candidate_sections = {
            'hardware': [
                'Hardware Requirements',
                'Hardware Specs',
                'Hardware Specifications',
                'Hardware Components',
                'Hardware Design',
                'Hardware',
                'Physical Requirements'
            ],
            'software': [
                'Software Requirements',
                'Software Specs',
                'Software Specifications',
                'Software Components',
                'Software Design',
                'Software',
                'Application Requirements'
            ],
            'firmware': [
                'Firmware Requirements',
                'Firmware Specs',
                'Firmware Specifications',
                'Firmware Components',
                'Firmware Design',
                'Firmware',
                'Embedded Software'
            ]
        }

    def _is_requirement_like(self, text: str, threshold: float = 60.0) -> bool:
        """Check if text is likely to be a requirement using fuzzy matching and keywords."""
        # Check for keywords
        has_keyword = any(keyword in text.lower() for keyword in self.requirement_keywords)
        
        # Check fuzzy match against common formats
        best_match = max(
            fuzz.partial_ratio(text.lower(), format.lower())
            for format in self.requirement_formats
        )
        
        return has_keyword or best_match >= threshold

    def _extract_priority(self, text: str) -> str:
        """Extract priority from text using fuzzy matching."""
        priorities = ['High', 'Medium', 'Low']
        words = text.split()
        
        for word in words:
            for priority in priorities:
                if fuzz.ratio(word.lower(), priority.lower()) >= 80:
                    return priority
        
        return 'Medium'  # Default priority

    def _clean_requirement_text(self, text: str) -> str:
        """Clean and normalize requirement text."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove common prefixes
        text = re.sub(r'^[•\-\*]\s*', '', text)
        # Remove common suffixes
        text = re.sub(r'\s*[;:]\s*$', '', text)
        return text.strip()

    def extract_text(self, pdf_path):
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    # Join hyphenated words and fix broken lines
                    page_text = page.extract_text()
                    page_text = re.sub(r'-\s*\n', '', page_text)  # Join hyphenated
                    page_text = re.sub(r'\s*\n\s*', ' ', page_text)  # Join broken lines
                    text += page_text + "\n"
                return text
        except Exception as e:
            return f"Error extracting text: {e}"
        # Configurable patterns for different PRD formats
        self.section_patterns = [
            r'\d+\.\d+\s*Hardware\s*Requirements',
            r'\d+\.\d+\s*Software\s*Requirements',
            r'\d+\.\d+\s*Firmware\s*Requirements',
            r'Technical\s*Requirements',
            r'System\s*Requirements',
            r'Features?\s*&\s*Requirements?'
        ]
        
        self.requirement_patterns = [
            # Pattern for "HW1 14 Digital I/O Pins High 6 should support PWM"
            r'^(?P<id>[A-Z]{2}\d+)\s+(?P<req>.+?)\s+(?P<pri>High|Medium|Low)\s+(?P<notes>.+)$',
            
            # Pattern for bullet points
            r'^-\s*(?P<req>.+?)\s*:\s*(?P<notes>.+)$',
            
            # Pattern for requirement sentences
            r'^(?:The system shall|Must|Should)\s+(?P<req>.+?)(?:\s*\((?P<notes>.+)\))?$'
        ]

    def locate_requirements_section(self, text: str) -> str:
        """Locate the requirements section in the text using fuzzy pattern matching.
        
        Args:
            text: The full text to search through
            
        Returns:
            The requirements section text if found, empty string otherwise
        """
        section_patterns = [
            r'4\.1\s*Hardware Requirements.*?(?=4\.2|5\.|UX|Technical Constraints)',
            r'Technical Requirements.*?(?=Dependencies|Milestones)',
            r'Requirements.*?(?=\d+\.\d+|\n\n|\Z)',
            r'\d+\.\d+\s*Hardware\s*Requirements.*?(?=\d+\.\d+|\n\n|\Z)',
            r'\d+\.\d+\s*Software\s*Requirements.*?(?=\d+\.\d+|\n\n|\Z)',
            r'\d+\.\d+\s*Firmware\s*Requirements.*?(?=\d+\.\d+|\n\n|\Z)',
            r'System\s*Requirements.*?(?=\d+\.\d+|\n\n|\Z)',
            r'Features?\s*&\s*Requirements?.*?(?=\d+\.\d+|\n\n|\Z)'
        ]
        
        for pattern in section_patterns:
            match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(0)
        return ""

    def extract_requirements(self, pdf_path: str) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        """Extract both hardware and software requirements from PDF"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                # First locate the requirements section
                scoped_text = self.locate_requirements_section(full_text)
                if not scoped_text:
                    return {"error": "No requirements section found in the document"}
                return self._process_text(scoped_text)
        except Exception as e:
            return {"error": f"Error processing PDF: {str(e)}"}

    def _process_text(self, text: str) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        """Process extracted text to find all types of requirements"""
        result = {
            "hardware": [],
            "software": [],
            "firmware": []
        }
        
        # Find all requirement sections
        sections = self._find_all_sections(text)
        
        for section_type, section_text in sections.items():
            if isinstance(section_text, str):
                result[section_type] = section_text
                continue
                
            # Extract requirements with multiple pattern attempts
            requirements = []
            for pattern in self.requirement_patterns:
                requirements.extend(self._extract_with_pattern(section_text, pattern))
                if requirements:
                    break
                    
            result[section_type] = requirements or f"No {section_type} requirements found matching expected patterns"
                
        return result

    def _find_all_sections(self, text: str) -> Dict[str, Union[str, str]]:
        """Locate all requirement sections in the text using fuzzy matching."""
        sections = {
            "hardware": "No hardware requirements section found",
            "software": "No software requirements section found",
            "firmware": "No firmware requirements section found"
        }
        
        # Split text into lines for processing
        lines = text.split('\n')
        current_section = None
        section_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try to match section header
            section_type, matched_header = self._fuzzy_match_header(line)
            
            if section_type:
                # If we were collecting content for a previous section, save it
                if current_section and section_content:
                    sections[current_section] = '\n'.join(section_content)
                    section_content = []
                
                current_section = section_type
                section_content.append(line)  # Include the header in the content
            elif current_section:
                section_content.append(line)
        
        # Save the last section if exists
        if current_section and section_content:
            sections[current_section] = '\n'.join(section_content)
        
        return sections

    def _fuzzy_match_header(self, text: str, threshold: float = 85.0) -> tuple[str, str]:
        """Match text against candidate section headers using fuzzy matching.
        
        Args:
            text: The text to match
            threshold: Minimum similarity score (0-100)
            
        Returns:
            Tuple of (section_type, matched_header) if match found, empty strings otherwise
        """
        text = text.lower().strip()
        best_score = 0
        best_match = ('', '')
        
        for section_type, headers in self.candidate_sections.items():
            for header in headers:
                score = fuzz.partial_ratio(text, header.lower())
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = (section_type, header)
        
        return best_match

    def _extract_with_pattern(self, text: str, pattern: str) -> List[Dict[str, str]]:
        """Extract requirements using a specific pattern with enhanced matching."""
        requirements = []
        
        # Split text into lines and process each line
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try exact pattern match first
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                req_dict = match.groupdict()
            else:
                # Try more flexible pattern
                flexible_pattern = r'(?P<id>HW\d+|FW\d+)?[:\s\-]*?(?P<req>.+?)[:\s\-]+(?P<pri>High|Medium|Low)?[:\s\-]*(?P<notes>.*)'
                match = re.match(flexible_pattern, line, re.IGNORECASE)
                if match:
                    req_dict = match.groupdict()
                else:
                    # If no pattern matches, check if it's requirement-like
                    if self._is_requirement_like(line):
                        req_dict = {
                            'id': '',
                            'req': line,
                            'pri': self._extract_priority(line),
                            'notes': ''
                        }
                    else:
                        continue
            
            # Clean and normalize the requirement
            req = {
                'id': req_dict.get('id', ''),
                'requirement': self._clean_requirement_text(req_dict.get('req', '')),
                'priority': req_dict.get('pri', self._extract_priority(line)),
                'notes': self._clean_requirement_text(req_dict.get('notes', ''))
            }
            
            # Only add if it looks like a requirement
            if self._is_requirement_like(req['requirement']):
                requirements.append(req)
        
        return requirements

    def generate_task_from_requirement(self, req: Dict[str, str]) -> Dict[str, str]:
        """Generate a task from a single requirement using Ollama.
        
        Args:
            req: Dictionary containing requirement information
            
        Returns:
            Dictionary containing the generated task
        """
        # Format the requirement into a sentence
        formatted_req = self._format_requirement_to_sentence(req)
        
        # Extract technical values for this requirement
        technical_values = self._extract_technical_values(formatted_req)
        
        # Create a focused prompt for this specific requirement
        prompt = f"""You are a task generator. Convert this requirement into a structured task.

Input Requirement: {formatted_req}

Technical Values Found:
{json.dumps(technical_values, indent=2)}

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
    "priority": "{req.get('priority', 'Medium')}",
    "assignee": "Hardware Engineer" if "HW" in req.get('id', '') else "Software Engineer",
    "due_date": "TBD"
}}"""

        # Generate task using Ollama
        generated_text = self._generate_with_api(prompt)
        
        # Extract JSON from the generated text
        try:
            json_match = re.search(r'\{[\s\S]*\}', generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for requirement {req.get('id', 'unknown')}: {str(e)}")
        except Exception as e:
            print(f"Error processing requirement {req.get('id', 'unknown')}: {str(e)}")
        
        return None

    def generate_tasks(self, context: str) -> List[Dict[str, str]]:
        """Generate tasks from individual requirements in the PRD using concurrent processing."""
        requirements = self.extract_requirements_from_text(context)
        
        if isinstance(requirements, str):
            print(f"Warning: {requirements}")
            return []
            
        tasks = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.generate_task_from_requirement, req) for req in requirements]
            for future in futures:
                try:
                    task = future.result()
                    if task:  # Only append if task was successfully generated
                        tasks.append(task)
                except Exception as e:
                    print(f"Task generation failed: {e}")
        
        return tasks

    def _format_requirement_to_sentence(self, req: Dict[str, str]) -> str:
        """Convert structured requirement to natural language sentence."""
        sentence = f"{req['id'] + ': ' if req['id'] else ''}The system requires {req['requirement'].lower()}"
        
        if req['notes'] and req['notes'].lower() not in ["none", "n/a", "na"]:
            notes = re.sub(r'[^\w\s.,!?0-9-]', '', req['notes'])
            sentence += f". Additional notes: {notes.lower()}"
            
        sentence += f". This is a {req['priority'].lower()}-priority task."
        return sentence

    def _extract_technical_values(self, text: str) -> dict:
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

    def _generate_with_api(self, prompt: str) -> str:
        """Generate text using Ollama."""
        try:
            # Initialize Ollama if not already done
            if not hasattr(self, 'llm'):
                from langchain_community.llms import Ollama
                from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
                callbacks = [StreamingStdOutCallbackHandler()]
                self.llm = Ollama(
                    model="mistral",
                    callbacks=callbacks,
                    temperature=0.7
                )
            
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            print(f"Ollama Error: {str(e)}")
            return ""