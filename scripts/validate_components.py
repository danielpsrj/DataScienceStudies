#!/usr/bin/env python3
"""
Component Usage Validator for Data Science Platform.

This script validates that concept pages properly use the reusable components
from app.components instead of reinventing UI elements.

Usage:
    python scripts/validate_components.py app/pages/03_📊_new_concept.py
"""

import re
import ast
import sys
from pathlib import Path
from typing import List, Set, Dict, Any


class ComponentUsageValidator:
    """Validator for component usage in concept pages."""

    # Components that should be used from app.components
    REQUIRED_COMPONENTS = {
        "theory_section": "Concept overview sections",
        "math_equation": "Mathematical formulations",
        "code_tabs": "Code examples in tabs",
        "display_applications": "Real-world applications (optional)",
        "display_pitfalls": "Common pitfalls (optional)",
        "display_references": "References in expander",
    }

    # Custom implementations to flag (anti-patterns)
    CUSTOM_IMPLEMENTATIONS = {
        r"def.*theory.*section": "Use theory_section from app.components",
        r"def.*math.*equation": "Use math_equation from app.components",
        r"def.*code.*tab": "Use code_tabs from app.components",
        r"st\.expander.*references.*manual": "Use display_references from app.components",
    }

    # Patterns for manual implementations that should use components
    MANUAL_PATTERNS = {
        "Manual theory section": r"st\.markdown.*theory.*st\.image",
        "Manual math equation": r"st\.latex.*st\.markdown.*variables",
        "Manual code tabs": r"tab1.*tab2.*st\.tabs.*code",
        "Manual applications": r"applications.*manual.*loop",
        "Manual pitfalls": r"pitfalls.*manual.*loop",
    }

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.content = ""
        self.imports: Set[str] = set()
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.suggestions: List[str] = []

    def load_file(self) -> bool:
        """Load and parse the Python file."""
        if not self.file_path.exists():
            self.errors.append(f"File not found: {self.file_path}")
            return False

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.content = f.read()

            # Parse imports using AST
            tree = ast.parse(self.content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        self.imports.add(f"{module}.{alias.name}")

            return True
        except Exception as e:
            self.errors.append(f"Error parsing file: {e}")
            return False

    def validate_component_imports(self):
        """Validate that required components are imported."""
        # Check for app.components import
        if not any("app.components" in imp for imp in self.imports):
            self.errors.append("Missing import from app.components")
            return

        # Check which components are imported
        imported_components = set()
        for imp in self.imports:
            if "app.components" in imp:
                # Extract component names
                if "." in imp:
                    component = imp.split(".")[-1]
                    imported_components.add(component)

        # Check for required components
        for component, description in self.REQUIRED_COMPONENTS.items():
            if component not in imported_components:
                # Check if it's being used anyway (might be imported differently)
                if component in self.content:
                    self.warnings.append(
                        f"Component '{component}' ({description}) might not be properly imported"
                    )
                else:
                    self.warnings.append(
                        f"Consider importing '{component}' ({description}) from app.components"
                    )

    def check_custom_implementations(self):
        """Check for custom implementations that should use components."""
        for pattern, message in self.CUSTOM_IMPLEMENTATIONS.items():
            if re.search(pattern, self.content, re.IGNORECASE):
                self.errors.append(f"Custom implementation detected: {message}")

    def check_manual_patterns(self):
        """Check for manual implementations that could use components."""
        for pattern_name, pattern in self.MANUAL_PATTERNS.items():
            if re.search(pattern, self.content, re.IGNORECASE):
                self.suggestions.append(
                    f"Manual {pattern_name.lower()} detected - consider using app.components"
                )

    def validate_component_usage(self):
        """Validate that components are actually used."""
        # Check for component usage in code
        for component in self.REQUIRED_COMPONENTS.keys():
            if component in self.content:
                # Check if it's actually called as a function
                if re.search(rf"{component}\(", self.content):
                    continue  # Component is being used
                else:
                    self.warnings.append(
                        f"Component '{component}' imported but not used"
                    )

    def check_for_reinvention(self):
        """Check for code that reinvents component functionality."""
        # Theory section reinvention
        if re.search(
            r"st\.title.*st\.markdown.*st\.image.*columns", self.content, re.DOTALL
        ):
            if "theory_section(" not in self.content:
                self.suggestions.append(
                    "Manual theory section implementation - use theory_section() component"
                )

        # Math equation reinvention
        if re.search(
            r"st\.latex.*st\.markdown.*variable.*explanation", self.content, re.DOTALL
        ):
            if "math_equation(" not in self.content:
                self.suggestions.append(
                    "Manual math equation - use math_equation() component"
                )

        # Code tabs reinvention
        if re.search(r"tab1.*tab2.*st\.code.*st\.tabs", self.content, re.DOTALL):
            if "code_tabs(" not in self.content:
                self.suggestions.append("Manual code tabs - use code_tabs() component")

    def validate_state_usage(self):
        """Validate state management usage."""
        if "from app.state import get_state" not in self.content:
            self.errors.append("Missing import: from app.state import get_state")

        if "get_state()" not in self.content:
            self.warnings.append(
                "State management not used - consider adding page tracking"
            )

    def analyze_structure(self):
        """Analyze overall structure for component usage patterns."""
        lines = self.content.split("\n")

        # Count component usage
        component_usage = {}
        for component in self.REQUIRED_COMPONENTS.keys():
            count = len(re.findall(rf"{component}\(", self.content))
            if count > 0:
                component_usage[component] = count

        if component_usage:
            print("\n📊 Component Usage Statistics:")
            for component, count in component_usage.items():
                print(f"  • {component}: {count} usage(s)")

        # Check for Streamlit anti-patterns
        sidebar_controls = len(
            re.findall(r"st\.sidebar\.(slider|selectbox|checkbox|button)", self.content)
        )
        if sidebar_controls > 0:
            self.warnings.append(
                f"Found {sidebar_controls} sidebar controls - ensure demo controls are in columns"
            )

    def run_validation(self) -> bool:
        """Run all validations."""
        if not self.load_file():
            return False

        self.validate_component_imports()
        self.check_custom_implementations()
        self.check_manual_patterns()
        self.validate_component_usage()
        self.check_for_reinvention()
        self.validate_state_usage()

        return len(self.errors) == 0

    def print_report(self):
        """Print validation report."""
        print(f"\n{'=' * 60}")
        print(f"Component Usage Validation: {self.file_path.name}")
        print(f"{'=' * 60}")

        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"  • {error}")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  • {warning}")

        if self.suggestions:
            print("\n💡 SUGGESTIONS:")
            for suggestion in self.suggestions:
                print(f"  • {suggestion}")

        if not self.errors and not self.warnings and not self.suggestions:
            print("\n✅ Excellent component usage!")

        self.analyze_structure()

        print(f"\n{'=' * 60}")
        print(
            f"Summary: {len(self.errors)} errors, {len(self.warnings)} warnings, {len(self.suggestions)} suggestions"
        )
        print(f"{'=' * 60}")

        return len(self.errors) == 0


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_components.py <page_file.py>")
        sys.exit(1)

    file_path = sys.argv[1]
    validator = ComponentUsageValidator(file_path)

    if validator.run_validation():
        print("✅ Validation completed successfully")
    else:
        print("❌ Validation failed")

    success = validator.print_report()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
