import unittest
import subprocess
import sys
import os
from os.path import abspath, dirname, join
from pathlib import Path
import tempfile
import shutil

testdir = dirname(abspath(str(__file__)))
root_dir = Path(join(testdir, "..", ".."))
examples_dir = Path(join(root_dir, "examples"))


class TestExamples(unittest.TestCase):
    """Test that all example scripts run without exceptions."""

    def setUp(self):
        """Set up test environment."""
        self.original_cwd = os.getcwd()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Copy examples directory to temp directory
        shutil.copytree(examples_dir, self.temp_dir / "examples")
        
        # Copy data directory to temp directory
        data_dir = Path(join(root_dir, "data"))
        if data_dir.exists():
            shutil.copytree(data_dir, self.temp_dir / "data", dirs_exist_ok=True)

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_all_examples_run_without_exceptions(self):
        """Test that all Python example scripts run without exceptions."""
        examples_path = self.temp_dir / "examples"
        
        # Find all Python files in the examples directory
        python_files = list(examples_path.glob("*.py"))
        
        # Filter out any non-example files (like __init__.py if it exists)
        example_files = [f for f in python_files if not f.name.startswith("__")]
        
        self.assertGreater(len(example_files), 0, "No Python example files found")
        
        for example_file in example_files:
            with self.subTest(example=example_file.name):
                # Run the example script from the examples directory
                try:
                    result = subprocess.run(
                        [sys.executable, example_file.name],
                        capture_output=True,
                        text=True,
                        timeout=300,  # 5 minute timeout
                        cwd=examples_path
                    )
                    
                    # Check if the script ran successfully
                    self.assertEqual(result.returncode, 0, 
                                   f"Script {example_file.name} failed with return code {result.returncode}.\n"
                                   f"STDOUT: {result.stdout}\n"
                                   f"STDERR: {result.stderr}")
                    
                    # Basic check that the script produced some output (stdout or stderr)
                    # This helps catch scripts that exit silently without doing anything
                    has_output = bool(result.stdout.strip()) or bool(result.stderr.strip())
                    if not has_output:
                        print(f"Warning: {example_file.name} produced no output")
                    
                except subprocess.TimeoutExpired:
                    self.fail(f"Example script {example_file.name} timed out after 5 minutes")
                except Exception as e:
                    self.fail(f"Unexpected error running example script {example_file.name}: {e}")

    def test_examples_directory_structure(self):
        """Test that the examples directory has the expected structure."""
        examples_path = self.temp_dir / "examples"
        
        # Check that examples directory exists and has Python files
        self.assertTrue(examples_path.exists(), "Examples directory does not exist")
        
        python_files = list(examples_path.glob("*.py"))
        self.assertGreater(len(python_files), 0, "No Python files found in examples directory")
        
        # Check for common expected files (optional - just for documentation)
        expected_files = [
            "generate_shell_segments.py",
            "run_shell_analysis.py", 
            "run_solid_analysis.py"
        ]
        
        for expected_file in expected_files:
            expected_path = examples_path / expected_file
            if expected_path.exists():
                print(f"Found expected example: {expected_file}")
            else:
                print(f"Expected example not found: {expected_file}")


if __name__ == "__main__":
    unittest.main()
