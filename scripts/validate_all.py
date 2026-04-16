#!/usr/bin/env python3
"""
Comprehensive Validator for Data Science Platform.

This script runs all validations for concept pages:
1. Page structure validation
2. Component usage validation
3. Test validation
4. Code quality checks

Usage:
    python scripts/validate_all.py app/pages/03_📊_new_concept.py
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


class ComprehensiveValidator:
    """Runs all validations for a concept page."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.success_count = 0
        self.total_checks = 0

    def run_command(self, command: List[str], check_name: str) -> Tuple[bool, str]:
        """Run a shell command and capture output."""
        self.total_checks += 1
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            self.success_count += 1
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            error_msg = f"{check_name} failed (exit code {e.returncode}):\n{e.stderr}"
            self.errors.append(error_msg)
            return False, e.stderr
        except Exception as e:
            error_msg = f"{check_name} error: {str(e)}"
            self.errors.append(error_msg)
            return False, str(e)

    def validate_page_structure(self) -> bool:
        """Run page structure validation."""
        print("\n🔍 Validating page structure...")
        success, output = self.run_command(
            ["python", "scripts/validate_page_structure.py", str(self.file_path)],
            "Page structure validation",
        )

        if success:
            print("✅ Page structure validation passed")
        else:
            print("❌ Page structure validation failed")

        return success

    def validate_component_usage(self) -> bool:
        """Run component usage validation."""
        print("\n🔍 Validating component usage...")
        success, output = self.run_command(
            ["python", "scripts/validate_components.py", str(self.file_path)],
            "Component usage validation",
        )

        if success:
            print("✅ Component usage validation passed")
        else:
            print("❌ Component usage validation failed")

        return success

    def run_tests(self) -> bool:
        """Run pytest on the project."""
        print("\n🔍 Running tests...")
        success, output = self.run_command(
            ["python", "-m", "pytest", "tests/", "-v"], "Test suite"
        )

        if success:
            print("✅ All tests passed")
        else:
            print("❌ Tests failed")

        return success

    def check_linting(self) -> bool:
        """Check code linting with ruff."""
        print("\n🔍 Checking linting...")
        success, output = self.run_command(["ruff", "check", "."], "Linting check")

        if success:
            print("✅ Linting passed")
        else:
            print("❌ Linting issues found")
            self.warnings.append("Linting issues - run 'ruff format .' to fix")

        return success

    def check_formatting(self) -> bool:
        """Check code formatting with ruff."""
        print("\n🔍 Checking formatting...")
        success, output = self.run_command(
            ["ruff", "format", "--check", "."], "Formatting check"
        )

        if success:
            print("✅ Formatting passed")
        else:
            print("❌ Formatting issues found")
            self.warnings.append("Formatting issues - run 'ruff format .' to fix")

        return success

    def check_imports(self) -> bool:
        """Check for unused imports."""
        print("\n🔍 Checking imports...")
        success, output = self.run_command(
            ["ruff", "check", ".", "--select", "F401"], "Unused imports check"
        )

        if success:
            print("✅ No unused imports")
        else:
            print("⚠️  Unused imports found")
            self.warnings.append("Unused imports detected")

        # This is a warning, not an error
        return True

    def validate_template_compliance(self) -> bool:
        """Check if page follows template structure."""
        print("\n🔍 Checking template compliance...")

        # Read the file
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            self.errors.append(f"Error reading file: {e}")
            return False

        # Check for required sections (simplified check)
        required_patterns = [
            ("Concept Overview", r"theory_section|st\.header.*Concept"),
            ("Interactive Demo", r"st\.header.*Interactive Demo"),
            ("Implementation Examples", r"st\.header.*Implementation Examples"),
            ("Real-World Applications", r"st\.header.*Real-World Applications"),
            ("Common Pitfalls", r"st\.header.*Common Pitfalls"),
            ("References", r"st\.header.*References"),
        ]

        missing_sections = []
        for section_name, pattern in required_patterns:
            if not re.search(pattern, content, re.IGNORECASE):
                missing_sections.append(section_name)

        if missing_sections:
            self.errors.append(f"Missing sections: {', '.join(missing_sections)}")
            print("❌ Template compliance failed")
            return False

        print("✅ Template compliance passed")
        return True

    def check_for_sidebar_anti_pattern(self) -> bool:
        """Check for sidebar usage in demo controls."""
        print("\n🔍 Checking for sidebar anti-patterns...")

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            self.errors.append(f"Error reading file: {e}")
            return False

        # Check for sidebar controls in demo context
        import re

        # Look for demo section
        demo_section_match = re.search(
            r"st\.header.*Interactive Demo.*?(?=st\.header|$)",
            content,
            re.DOTALL | re.IGNORECASE,
        )

        if demo_section_match:
            demo_section = demo_section_match.group(0)
            if "st.sidebar" in demo_section:
                self.errors.append("Demo controls should not be in sidebar")
                print("❌ Sidebar anti-pattern detected")
                return False

        print("✅ No sidebar anti-patterns")
        return True

    def run_all_validations(self) -> bool:
        """Run all validations."""
        print(f"\n{'=' * 60}")
        print(f"Comprehensive Validation: {self.file_path.name}")
        print(f"{'=' * 60}")

        # Run validations
        validations = [
            self.validate_page_structure,
            self.validate_component_usage,
            self.run_tests,
            self.check_linting,
            self.check_formatting,
            self.check_imports,
            self.validate_template_compliance,
            self.check_for_sidebar_anti_pattern,
        ]

        results = []
        for validation in validations:
            try:
                results.append(validation())
            except Exception as e:
                self.errors.append(f"Validation error in {validation.__name__}: {e}")
                results.append(False)

        # Print summary
        print(f"\n{'=' * 60}")
        print("VALIDATION SUMMARY")
        print(f"{'=' * 60}")

        print(f"\n📊 Results: {self.success_count}/{self.total_checks} checks passed")

        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                # Print first line of each error for brevity
                error_lines = error.split("\n")
                print(f"  • {error_lines[0]}")
                if len(error_lines) > 1:
                    print(f"    ... ({len(error_lines) - 1} more lines)")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  • {warning}")

        if not self.errors and not self.warnings:
            print("\n🎉 ALL VALIDATIONS PASSED!")
            print("The page is ready for commit.")

        print(f"\n{'=' * 60}")

        return len(self.errors) == 0


# Import regex module for template compliance check
import re


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_all.py <page_file.py>")
        print("\nExample:")
        print("  python scripts/validate_all.py app/pages/03_📊_new_concept.py")
        sys.exit(1)

    file_path = sys.argv[1]

    # Check if file exists
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    validator = ComprehensiveValidator(file_path)
    success = validator.run_all_validations()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
