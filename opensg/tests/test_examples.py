# This is a test to ensure all of the examples run.
import os
import sys
import unittest
from os import listdir
from os.path import abspath, dirname, isfile, join
from subprocess import call

testdir = dirname(abspath(str(__file__)))
examplesdir = join(testdir, "..", "..", "examples")


class TestExamples(unittest.TestCase):
    """Test that all example scripts run without exceptions."""

    @classmethod
    def setUpClass(cls):
        import opensg
        cls.opensg = opensg

    @classmethod
    def tearDownClass(cls):
        pass
    
    def test_that_examples_run(self):
        """Test that all Python example scripts run without exceptions."""
        cwd = os.getcwd()
        os.chdir(examplesdir)
        
        example_files = [
            f
            for f in listdir(examplesdir)
            if isfile(join(examplesdir, f))
            and f.endswith(".py")
            and not f.startswith("test")
        ]
        
        flag = 0
        failed_examples = []
        
        for f in example_files:
            tmp_flag = call([sys.executable, join(examplesdir, f)])
            print(f, tmp_flag)
            if tmp_flag == 1:
                failed_examples.append(f)
                flag = 1
        
        os.chdir(cwd)
        
        if len(failed_examples) > 0:
            print("failed examples: {0}".format(failed_examples))
        
        self.assertEqual(flag, 0)


if __name__ == "__main__":
    unittest.main()
