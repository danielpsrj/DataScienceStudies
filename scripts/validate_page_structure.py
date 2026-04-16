#!/usr/bin/env python3
"""
Page Structure Validator for Data Science Platform.

This script validates that concept pages follow the required 6-section structure:
1. Concept Overview
2. Interactive Demo
3. Implementation Examples
4. Real-World Applications
5. Common Pitfalls & Fixes
6. References & Further Reading

Usage:
    python scripts/validate_page_structure.py app/pages/03_📊_new_concept.py
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple


class PageStructureValidator:
    """Validator for concept page structure."""

    # Required section headers (in order)
    REQUIRED_SECTIONS = [
        ("Concept Overview", ["theory_section", "st\\.header\\(.*Concept.*"]),
        (
            "Interactive Demo",
            ["st\\.header\\(.*Interactive Demo.*", "demo_col1.*demo_col2"],
        ),
        (
            "Implementation Examples",
            ["st\\.header\\(.*Implementation Examples.*", "code_tabs"],
        ),
        (
            "Real-World Applications",
            [
                "st\\.header\\(.*Real-World Applications.*",
                "applications_data",
                "app_tabs.*st\\.tabs",
            ],
        ),
        (
            "Common Pitfalls & Fixes",
            [
                "st\\.header\\(.*Common Pitfalls.*",
                "pitfalls_data",
                "pitfall_tabs.*st\\.tabs",
            ],
        ),
        (
            "References & Further Reading",
            ["st\\.header\\(.*References.*", "st\\.expander\\(.*references.*"],
        ),
    ]

    # Anti-patterns to check for
    ANTI_PATTERNS = [
        (
            r"st\\.sidebar\\.(slider|selectbox|checkbox|button).*demo",
            "Demo controls should be in columns, not sidebar",
        ),
        (r"with st\\.sidebar:", "Avoid using sidebar for demo controls"),
        (
            r"st\\.header\\(.*\\)\\s*# Only header, no content",
            "Sections should have content after header",
        ),
    ]

    # Required imports
    REQUIRED_IMPORTS = [
        "from app.components import",
        "from app.state import get_state",
        "import streamlit as st",
    ]

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.content = ""
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def load_file(self) -> bool:
        """Load and validate file exists."""
        if not self.file_path.exists():
            self.errors.append(f"File not found: {self.file_path}")
            return False

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.content = f.read()
            return True
        except Exception as e:
            self.errors.append(f"Error reading file: {e}")
            return False

    def validate_imports(self):
        """Validate required imports are present."""
        for required_import in self.REQUIRED_IMPORTS:
            if required_import not in self.content:
                self.errors.append(f"Missing required import: {required_import}")

    def validate_section_structure(self):
        """Validate the 6-section structure."""
        lines = self.content.split("\n")

        # Find all section headers
        section_headers = []
        for i, line in enumerate(lines):
            if "st.header(" in line or "st.title(" in line:
                # Extract header text
                match = re.search(r'st\.(header|title)\(["\'](.*?)["\']\)', line)
                if match:
                    section_headers.append((i, match.group(2)))

        # Check for required sections
        found_sections = []
        for section_name, patterns in self.REQUIRED_SECTIONS:
            found = False
            for pattern in patterns:
                if re.search(pattern, self.content, re.IGNORECASE):
                    found = True
                    break

            if found:
                found_sections.append(section_name)
            else:
                self.errors.append(f"Missing section: {section_name}")

        # Check section order (basic check)
        if len(found_sections) >= 2:
            # Verify sections appear in correct relative order
            section_positions = []
            for section_name, patterns in self.REQUIRED_SECTIONS:
                for pattern in patterns:
                    match = re.search(pattern, self.content, re.IGNORECASE)
                    if match:
                        section_positions.append((section_name, match.start()))
                        break

            # Sort by position and check order
            section_positions.sort(key=lambda x: x[1])
            ordered_sections = [name for name, _ in section_positions]

            # Check if order matches required order
            for i, (found, required) in enumerate(
                zip(ordered_sections, [name for name, _ in self.REQUIRED_SECTIONS])
            ):
                if found != required:
                    self.warnings.append(
                        f"Section order issue: Expected '{required}' but found '{found}' at position {i}"
                    )

    def check_anti_patterns(self):
        """Check for anti-patterns."""
        for pattern, message in self.ANTI_PATTERNS:
            if re.search(pattern, self.content):
                self.errors.append(f"Anti-pattern detected: {message}")

    def validate_demo_controls(self):
        """Validate demo controls are in columns, not sidebar."""
        # Check for demo columns pattern
        if not re.search(r"demo_col1.*demo_col2.*st\.columns", self.content):
            self.warnings.append("Demo controls might not be in columns pattern")

        # Check for sidebar usage in demo context
        demo_section = self._extract_section("Interactive Demo")
        if demo_section and "st.sidebar" in demo_section:
            self.errors.append("Demo controls should not be in sidebar")

    def validate_tab_structures(self):
        """Validate applications and pitfalls are in tabs."""
        # Check applications tabs
        if "app_tabs = st.tabs" not in self.content:
            self.warnings.append("Applications might not be using tabs structure")

        # Check pitfalls tabs
        if "pitfall_tabs = st.tabs" not in self.content:
            self.warnings.append("Pitfalls might not be using tabs structure")

    def validate_references_expander(self):
        """Validate references are in expander."""
        if not re.search(r"with st\.expander.*references", self.content, re.IGNORECASE):
            self.warnings.append("References might not be inside expander")

    def _extract_section(self, section_name: str) -> str:
        """Extract a section from the content."""
        # Simple extraction based on headers
        lines = self.content.split("\n")
        in_section = False
        section_lines = []

        for line in lines:
            if f'"{section_name}"' in line or f"'{section_name}'" in line:
                in_section = True
                continue

            if in_section:
                # Check if we hit next major section
                if any(
                    f'"{next_section}"' in line or f"'{next_section}'" in line
                    for next_section, _ in self.REQUIRED_SECTIONS
                    if next_section != section_name
                ):
                    break
                section_lines.append(line)

        return "\n".join(section_lines)

    def run_validation(self) -> bool:
        """Run all validations."""
        if not self.load_file():
            return False

        self.validate_imports()
        self.validate_section_structure()
        self.check_anti_patterns()
        self.validate_demo_controls()
        self.validate_tab_structures()
        self.validate_references_expander()

        return len(self.errors) == 0

    def print_report(self):
        """Print validation report."""
        print(f"\n{'=' * 60}")
        print(f"Page Structure Validation: {self.file_path.name}")
        print(f"{'=' * 60}")

        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"  • {error}")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  • {warning}")

        if not self.errors and not self.warnings:
            print("\n✅ All validations passed!")

        print(f"\n{'=' * 60}")
        print(f"Summary: {len(self.errors)} errors, {len(self.warnings)} warnings")
        print(f"{'=' * 60}")

        return len(self.errors) == 0


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_page_structure.py <page_file.py>")
        sys.exit(1)

    file_path = sys.argv[1]
    validator = PageStructureValidator(file_path)

    if validator.run_validation():
        print("✅ Validation completed successfully")
    else:
        print("❌ Validation failed")

    success = validator.print_report()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
