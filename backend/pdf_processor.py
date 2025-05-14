import pdfplumber
import re
from typing import List, Dict, Optional, Union
class PRDExtractor:
    def __init__(self):
        # Initialize patterns before other methods
        self.section_patterns = []
        self.requirement_patterns = []

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

    def extract_requirements(self, pdf_path: str) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        """Extract both hardware and software requirements from PDF"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                return self._process_text(full_text)
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
        """Locate all requirement sections in the text"""
        sections = {
            "hardware": "No hardware requirements section found",
            "software": "No software requirements section found",
            "firmware": "No firmware requirements section found"
        }
        
        current_pos = 0
        while current_pos < len(text):
            # Find the next section
            next_section = None
            next_section_type = None
            next_section_start = len(text)
            
            for pattern in self.section_patterns:
                match = re.search(pattern, text[current_pos:], re.IGNORECASE)
                if match and match.start() + current_pos < next_section_start:
                    next_section = match
                    next_section_start = match.start() + current_pos
                    if "hardware" in pattern.lower():
                        next_section_type = "hardware"
                    elif "software" in pattern.lower():
                        next_section_type = "software"
                    elif "firmware" in pattern.lower():
                        next_section_type = "firmware"
            
            if next_section is None:
                break
                
            # Find the end of this section
            section_start = next_section_start
            section_end = len(text)
            end_match = re.search(r'\n\d+\.\d+\s+', text[section_start + next_section.end():])
            if end_match:
                section_end = section_start + next_section.end() + end_match.start()
            
            # Store the section
            if next_section_type:
                sections[next_section_type] = text[section_start:section_end]
            
            current_pos = section_end
            
        return sections

    def _extract_with_pattern(self, text: str, pattern: str) -> List[Dict[str, str]]:
        """Extract requirements using a specific pattern"""
        requirements = []
        for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
            req = {
                'id': match.groupdict().get('id', ''),
                'requirement': match.group('req').strip(),
                'priority': match.groupdict().get('pri', 'Medium'),
                'notes': match.groupdict().get('notes', '').strip()
            }
            requirements.append(req)
        return requirements