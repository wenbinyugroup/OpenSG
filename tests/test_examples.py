import os
import sys
import unittest
import pytest
from os import listdir
from os.path import isfile, join
from subprocess import call
from tests import root_dir

examples_dir = str(root_dir / "examples")


@pytest.mark.time_consuming
class TestExamples(unittest.TestCase):
    def test_examples_run(self):
        cwd = os.getcwd()
        os.chdir(examples_dir)
        example_files = [
            f
            for f in listdir(examples_dir)
            if isfile(join(examples_dir, f)) and f.endswith(".py")
        ]
        failed = []
        for f in example_files:
            rc = call([sys.executable, join(examples_dir, f)])
            print(f, rc)
            if rc != 0:
                failed.append(f)
        os.chdir(cwd)
        if failed:
            print("Failed examples: {0}".format(failed))
        self.assertEqual(len(failed), 0)


if __name__ == "__main__":
    unittest.main()
