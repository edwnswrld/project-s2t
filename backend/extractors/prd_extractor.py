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
        
        # Hardware-specific keywords for better identification
        self.hardware_keywords = [
            'circuit', 'pcb', 'schematic', 'layout', 'component',
            'bom', 'bill of materials', 'footprint', 'gerber',
            'esd', 'emi', 'signal integrity', 'power integrity',
            'voltage', 'current', 'resistance', 'capacitance',
            'inductance', 'impedance', 'solder', 'assembly',
            'test point', 'connector', 'header', 'via', 'trace',
            'copper', 'layer', 'drill', 'pad', 'silkscreen',
            'solder mask', 'plated through hole', 'surface mount',
            'through hole', 'smd', 'tht', 'dip', 'qfp', 'bga',
            'microcontroller', 'processor', 'ic', 'integrated circuit',
            'oscillator', 'crystal', 'resistor', 'capacitor', 'inductor',
            'diode', 'transistor', 'mosfet', 'opamp', 'operational amplifier',
            'adc', 'dac', 'pwm', 'uart', 'spi', 'i2c', 'gpio',
            'power supply', 'voltage regulator', 'battery', 'charger',
            'sensor', 'actuator', 'motor', 'servo', 'led', 'display',
            'button', 'switch', 'potentiometer', 'encoder', 'relay'
        ]
        
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

        # Enhanced patterns for sub-requirements
        self.requirement_patterns = [
            # Pattern for "HW1 14 Digital I/O Pins High 6 should support PWM"
            r'^(?P<id>[A-Z]{2}\d+)\s+(?P<req>.+?)\s+(?P<pri>High|Medium|Low)\s+(?P<notes>.+)$',
            
            # Pattern for hardware component specs
            r'^(?P<id>[A-Z]{2}\d+)\s+(?P<component>.+?)\s+'
            r'(?P<value>\d+[a-zA-Z]*)\s+(?P<tolerance>±?\d+%?)\s*'
            r'(?P<notes>.+)?$',
            
            # Pattern for test requirements
            r'^(?P<id>TEST-\d+)\s+Verify\s+(?P<parameter>.+?)\s+'
            r'is\s+(?P<condition>.+?)\s+with\s+(?P<method>.+)$',
            
            # Pattern for sub-requirements like "6 should support PWM"
            r'^(?P<sub_id>\d+)\s+(?P<req>should|must|shall|will|needs to)\s+(?P<action>.+?)(?:\s*\((?P<notes>.+)\))?$',
            
            # Pattern for bullet points
            r'^-\s*(?P<req>.+?)\s*:\s*(?P<notes>.+)$',
            
            # Pattern for requirement sentences
            r'^(?:The system shall|Must|Should)\s+(?P<req>.+?)(?:\s*\((?P<notes>.+)\))?$'
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

    def _is_hardware_requirement(self, text: str) -> bool:
        """Check if requirement is hardware-related."""
        text = text.lower()
        return any(keyword in text for keyword in self.hardware_keywords)

    def _is_requirement_like(self, text: str, threshold: float = 60.0) -> bool:
        """Check if text is likely to be a requirement using fuzzy matching and keywords."""
        # Check for keywords
        has_keyword = any(keyword in text.lower() for keyword in self.requirement_keywords)
        
        # Check fuzzy match against common formats
        best_match = max(
            fuzz.partial_ratio(text.lower(), format.lower())
            for format in self.requirement_formats
        )
        
        # Check if it's a hardware requirement
        is_hardware = self._is_hardware_requirement(text)
        
        return has_keyword or best_match >= threshold or is_hardware

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
                    
                    # Handle sub-requirements
                    if 'sub_id' in req_dict:
                        req_dict['id'] = f"SUB{req_dict['sub_id']}"
                        req_dict['requirement'] = f"{req_dict['req']} {req_dict['action']}"
                        del req_dict['sub_id']
                        del req_dict['action']
                    elif 'req' in req_dict:
                        req_dict['requirement'] = req_dict.pop('req')
                    
                    # Ensure all required fields are present
                    if 'id' not in req_dict:
                        req_dict['id'] = ''
                    if 'priority' not in req_dict:
                        req_dict['priority'] = self._extract_priority(line)
                    if 'notes' not in req_dict:
                        req_dict['notes'] = ''
                    
                    # Clean the requirement text
                    req_dict['requirement'] = self._clean_requirement_text(req_dict['requirement'])
                    
                    # Only add if it looks like a requirement
                    if self._is_requirement_like(req_dict['requirement']):
                        requirements.append(req_dict)
                    break
        
        return requirements

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

    def generate_task_from_requirement(self, req: Dict[str, str]) -> Dict[str, str]:
        """Generate a task from a single requirement using Ollama."""
        # Format the requirement into a sentence
        formatted_req = self._format_requirement_to_sentence(req)
        
        # Extract technical values for this requirement
        technical_values = self._extract_technical_values(formatted_req)
        
        # Determine if this is a hardware requirement
        is_hardware = self._is_hardware_requirement(formatted_req)
        
        # Create a focused prompt for this specific requirement
        prompt = f"""You are a task generator. Convert this requirement into a structured task.

Input Requirement: {formatted_req}

Technical Values Found:
{json.dumps(technical_values, indent=2)}

Requirement Type: {"Hardware" if is_hardware else "Software"}

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
    "assignee": "Hardware Engineer" if {"Hardware" if is_hardware else "Software"} else "Software Engineer",
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