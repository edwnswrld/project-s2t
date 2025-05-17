"""
Example perfect outputs for the Spec-to-Sprint AI tool.
These examples demonstrate the expected format and quality of task generation.
"""

# Example 1: Hardware Requirement with Technical Specifications
HARDWARE_REQ_EXAMPLE = {
    "input": """
    HW1: The system shall implement ESD protection on all external connectors.
    Voltage rating: ±8kV contact discharge, ±15kV air discharge.
    Test method: IEC 61000-4-2.
    Priority: High
    """,
    "expected_output": {
        "title": "Implement ESD Protection for External Connectors",
        "description": "Design and implement ESD protection circuitry for all external connectors meeting IEC 61000-4-2 standards with ±8kV contact and ±15kV air discharge ratings.",
        "tasks": [
            "Review current connector design and identify ESD protection requirements",
            "Select appropriate ESD protection components (TVS diodes, etc.)",
            "Design ESD protection circuit layout",
            "Create test plan for IEC 61000-4-2 compliance",
            "Document ESD protection implementation"
        ],
        "acceptance_criteria": [
            "ESD protection meets ±8kV contact discharge rating",
            "ESD protection meets ±15kV air discharge rating",
            "Compliance with IEC 61000-4-2 verified through testing",
            "Documentation includes component selection rationale and test results"
        ],
        "priority": "High",
        "assignee": "Hardware Engineer",
        "due_date": "TBD (2 weeks)"
    }
}

# Example 2: Software Requirement with Performance Metrics
SOFTWARE_REQ_EXAMPLE = {
    "input": """
    SW1: The firmware shall support over-the-air (OTA) updates.
    Maximum update time: 5 minutes.
    Must maintain system stability during update.
    Priority: Medium
    """,
    "expected_output": {
        "title": "Implement OTA Update System",
        "description": "Develop and implement over-the-air update functionality with 5-minute maximum update time while ensuring system stability throughout the process.",
        "tasks": [
            "Design OTA update protocol and architecture",
            "Implement secure update verification mechanism",
            "Create update process with rollback capability",
            "Develop update progress monitoring system",
            "Write update documentation and user guide"
        ],
        "acceptance_criteria": [
            "OTA updates complete within 5 minutes",
            "System remains stable during update process",
            "Failed updates can be rolled back automatically",
            "Update progress is visible to users",
            "Documentation covers update process and troubleshooting"
        ],
        "priority": "Medium",
        "assignee": "Firmware Engineer",
        "due_date": "TBD (3 weeks)"
    }
}

# Example 3: Table Format Requirement
TABLE_REQ_EXAMPLE = {
    "input": """
    Component    Specification    Priority
    MCU         ARM Cortex-M4    High
    Flash       512KB           High
    RAM         128KB           High
    Clock       100MHz          Medium
    """,
    "expected_output": {
        "title": "Select and Implement MCU Configuration",
        "description": "Implement system using ARM Cortex-M4 MCU with 512KB Flash and 128KB RAM, operating at 100MHz.",
        "tasks": [
            "Select specific ARM Cortex-M4 MCU model",
            "Configure MCU for 100MHz operation",
            "Set up Flash and RAM memory regions",
            "Implement power management for MCU",
            "Create MCU initialization code"
        ],
        "acceptance_criteria": [
            "MCU successfully operates at 100MHz",
            "512KB Flash memory properly configured and accessible",
            "128KB RAM properly configured and accessible",
            "Power consumption within specifications",
            "Initialization code documented and tested"
        ],
        "priority": "High",
        "assignee": "Hardware Engineer",
        "due_date": "TBD (2 weeks)"
    }
}

# Example 4: Complex Requirement with Multiple Dependencies
COMPLEX_REQ_EXAMPLE = {
    "input": """
    HW2: The power supply shall provide 5V and 3.3V outputs.
    Efficiency > 90% at full load.
    Must include overcurrent protection.
    Temperature monitoring required.
    Priority: High
    """,
    "expected_output": {
        "title": "Design and Implement Dual-Output Power Supply System",
        "description": "Design and implement a power supply system providing 5V and 3.3V outputs with >90% efficiency, overcurrent protection, and temperature monitoring.",
        "tasks": [
            "Select power supply topology and components",
            "Design power supply circuit with efficiency optimization",
            "Implement overcurrent protection circuit",
            "Design temperature monitoring system",
            "Create test plan for efficiency and protection features",
            "Document power supply design and test results"
        ],
        "acceptance_criteria": [
            "Power supply provides stable 5V and 3.3V outputs",
            "Efficiency exceeds 90% at full load",
            "Overcurrent protection trips at specified threshold",
            "Temperature monitoring provides accurate readings",
            "All protection features verified through testing",
            "Documentation includes efficiency curves and protection thresholds"
        ],
        "priority": "High",
        "assignee": "Hardware Engineer",
        "due_date": "TBD (3 weeks)"
    }
}

# Example 5: Documentation Requirement
DOC_REQ_EXAMPLE = {
    "input": """
    DOC1: Create comprehensive user manual.
    Include troubleshooting guide.
    Add safety warnings.
    Priority: Medium
    """,
    "expected_output": {
        "title": "Create Comprehensive User Manual with Safety Guidelines",
        "description": "Develop a comprehensive user manual including troubleshooting guide and safety warnings for the system.",
        "tasks": [
            "Create user manual structure and outline",
            "Write system overview and features section",
            "Develop step-by-step user instructions",
            "Create troubleshooting guide with common issues",
            "Add safety warnings and precautions",
            "Review and validate all documentation"
        ],
        "acceptance_criteria": [
            "User manual covers all system features",
            "Troubleshooting guide includes common issues and solutions",
            "Safety warnings are clear and prominent",
            "Documentation is reviewed by technical team",
            "User manual is tested with sample users"
        ],
        "priority": "Medium",
        "assignee": "Technical Writer",
        "due_date": "TBD (2 weeks)"
    }
}

def get_example_outputs():
    """Return all example outputs for testing and validation."""
    return {
        "hardware": HARDWARE_REQ_EXAMPLE,
        "software": SOFTWARE_REQ_EXAMPLE,
        "table": TABLE_REQ_EXAMPLE,
        "complex": COMPLEX_REQ_EXAMPLE,
        "documentation": DOC_REQ_EXAMPLE
    } 