#tests that intital requirements are expanded into a natural English sentence 
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_pipeline import format_requirement_to_sentence

def parse_requirement_string(req_str: str) -> dict:
    """Parse a requirement string into a dictionary format.
    
    Args:
        req_str: String in format "ID    Requirement    Priority    Notes"
        
    Returns:
        Dictionary with id, requirement, priority, and notes
    """
    # Split by either tab or multiple spaces
    parts = [part.strip() for part in req_str.split() if part.strip()]
    
    if len(parts) >= 4:
        return {
            'id': parts[0],
            'requirement': ' '.join(parts[1:-2]),  # Join middle parts as requirement
            'priority': parts[-2],
            'notes': parts[-1]
        }
    return {
        'id': '',
        'requirement': req_str,
        'priority': 'Medium',
        'notes': ''
    }

def test_requirement_expansion():
    # Test cases with different formats
    test_cases = [
        # Tab-separated format
        "HW1\t14 Digital I/O Pins\tHigh\t6 should support PWM",
        # Space-separated format
        "HW1    14 Digital I/O Pins    High    6 should support PWM",
        # Mixed format
        "HW1\t14 Digital I/O Pins    High\t6 should support PWM",
        # Another example with different content
        "HW2\t8 Analog Input Pins\tMedium\t2 should be high precision",
        # Example with no notes
        "HW3\t32KB Flash Memory\tHigh\tnone"
    ]

    print("\nTesting Requirement Expansion")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print("-" * 40)
        print("Input Requirement:")
        print(f'"{test_case}"')
        print("\nExpanded Sentence:")
        # Parse the string into a dictionary before passing to format_requirement_to_sentence
        req_dict = parse_requirement_string(test_case)
        expanded = format_requirement_to_sentence(req_dict)
        print(f'"{expanded}"')
        print("-" * 40)

if __name__ == "__main__":
    test_requirement_expansion() 