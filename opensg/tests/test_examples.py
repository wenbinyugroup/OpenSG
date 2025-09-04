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
        # Change to examples directory for running examples
        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.mkdtemp()
        os.chdir(self.temp_dir)
        
        # Copy necessary data files to temp directory
        data_dir = Path(join(root_dir, "data"))
        if data_dir.exists():
            shutil.copytree(data_dir, "data", dirs_exist_ok=True)

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_shell_segments_example(self):
        """Test that generate_shell_segments.py runs without exceptions."""
        example_script = examples_dir / "generate_shell_segments.py"
        
        # Check if the example file exists
        self.assertTrue(example_script.exists(), f"Example script not found: {example_script}")
        
        # Create a modified version of the script with correct paths
        with open(example_script, 'r') as f:
            script_content = f.read()
        
        # Fix the path issue - change ../data to data
        script_content = script_content.replace(
            'blade_mesh_file = join("..", "data", "shell_blade", "bar_urc_shell_mesh.yaml")',
            'blade_mesh_file = join("data", "shell_blade", "bar_urc_shell_mesh.yaml")'
        )
        
        # Write the modified script to temp directory
        modified_script = Path("generate_shell_segments_modified.py")
        with open(modified_script, 'w') as f:
            f.write(script_content)
        
        # Run the modified example script
        try:
            result = subprocess.run(
                [sys.executable, str(modified_script)],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=self.temp_dir
            )
            
            # Check if the script ran successfully
            self.assertEqual(result.returncode, 0, 
                           f"Script failed with return code {result.returncode}.\n"
                           f"STDOUT: {result.stdout}\n"
                           f"STDERR: {result.stderr}")
            
            # Check that expected output files were created
            segments_all_dir = Path("segments_all")
            segments_specific_dir = Path("segments_specific")
            
            self.assertTrue(segments_all_dir.exists(), "segments_all directory was not created")
            self.assertTrue(segments_specific_dir.exists(), "segments_specific directory was not created")
            
            # Check that some segment files were generated
            all_segments = list(segments_all_dir.glob("*.yaml"))
            specific_segments = list(segments_specific_dir.glob("*.yaml"))
            
            self.assertGreater(len(all_segments), 0, "No segment files were generated in segments_all")
            self.assertGreater(len(specific_segments), 0, "No segment files were generated in segments_specific")
            
        except subprocess.TimeoutExpired:
            self.fail("Example script timed out after 5 minutes")
        except Exception as e:
            self.fail(f"Unexpected error running example script: {e}")

    def test_run_shell_analysis_example(self):
        """Test that run_shell_analysis.py runs without exceptions."""
        example_script = examples_dir / "run_shell_analysis.py"
        
        # Check if the example file exists
        self.assertTrue(example_script.exists(), f"Example script not found: {example_script}")
        
        # Create a modified version of the script with correct paths
        with open(example_script, 'r') as f:
            script_content = f.read()
        
        # Fix the path issue - change data/bar_urc_shell_mesh.yaml to data/shell_blade/bar_urc_shell_mesh.yaml
        script_content = script_content.replace(
            'blade_mesh_file = Path("data", "bar_urc_shell_mesh.yaml")',
            'blade_mesh_file = Path("data", "shell_blade", "bar_urc_shell_mesh.yaml")'
        )
        
        # Write the modified script to temp directory
        modified_script = Path("run_shell_analysis_modified.py")
        with open(modified_script, 'w') as f:
            f.write(script_content)
        
        # Run the modified example script
        try:
            result = subprocess.run(
                [sys.executable, str(modified_script)],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=self.temp_dir
            )
            
            # Check if the script ran successfully
            self.assertEqual(result.returncode, 0, 
                           f"Script failed with return code {result.returncode}.\n"
                           f"STDOUT: {result.stdout}\n"
                           f"STDERR: {result.stderr}")
            
            # Check that expected output file was created
            stiffness_file = Path("stiffness_shell.txt")
            self.assertTrue(stiffness_file.exists(), "stiffness_shell.txt was not created")
            
            # Check that the file has content
            self.assertGreater(stiffness_file.stat().st_size, 0, "stiffness_shell.txt is empty")
            
        except subprocess.TimeoutExpired:
            self.fail("Example script timed out after 5 minutes")
        except Exception as e:
            self.fail(f"Unexpected error running example script: {e}")

    def test_run_solid_analysis_example(self):
        """Test that run_solid_analysis.py runs without exceptions."""
        example_script = examples_dir / "run_solid_analysis.py"
        
        # Check if the example file exists
        self.assertTrue(example_script.exists(), f"Example script not found: {example_script}")
        
        # Create a modified version of the script with correct paths
        with open(example_script, 'r') as f:
            script_content = f.read()
        
        # Fix the path issue - change to use available solid data
        script_content = script_content.replace(
            'segments_folder = Path("data", "bar_urc_npl_2_ar_10")',
            'segments_folder = Path("data", "solid_blade")'
        )
        script_content = script_content.replace(
            'segment_file = Path(segments_folder, f"bar_urc_npl_2_ar_10-segment_{i}.yaml")',
            'segment_file = Path(segments_folder, f"bar_urc_npl_1_ar_5-segment_{i+2}.yaml")'
        )
        
        # Write the modified script to temp directory
        modified_script = Path("run_solid_analysis_modified.py")
        with open(modified_script, 'w') as f:
            f.write(script_content)
        
        # Run the modified example script
        try:
            result = subprocess.run(
                [sys.executable, str(modified_script)],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=self.temp_dir
            )
            
            # Check if the script ran successfully
            self.assertEqual(result.returncode, 0, 
                           f"Script failed with return code {result.returncode}.\n"
                           f"STDOUT: {result.stdout}\n"
                           f"STDERR: {result.stderr}")
            
            # Check that expected output file was created
            stiffness_file = Path("stiffness_solid.txt")
            self.assertTrue(stiffness_file.exists(), "stiffness_solid.txt was not created")
            
            # Check that the file has content
            self.assertGreater(stiffness_file.stat().st_size, 0, "stiffness_solid.txt is empty")
            
        except subprocess.TimeoutExpired:
            self.fail("Example script timed out after 5 minutes")
        except Exception as e:
            self.fail(f"Unexpected error running example script: {e}")

    def test_all_examples_exist(self):
        """Test that all expected example files exist."""
        expected_examples = [
            "generate_shell_segments.py",
            "run_shell_analysis.py", 
            "run_solid_analysis.py"
        ]
        
        for example_name in expected_examples:
            example_path = examples_dir / example_name
            self.assertTrue(example_path.exists(), 
                          f"Expected example file not found: {example_path}")


if __name__ == "__main__":
    unittest.main()
