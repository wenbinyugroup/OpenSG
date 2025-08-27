import unittest
from os.path import abspath, dirname, join
from pathlib import Path
import numpy as np
import opensg
from opensg.mesh.segment import ShellSegmentMesh
import filecmp
import shutil
import os

testdir = dirname(abspath(str(__file__)))
root_dir = Path(join(testdir, "..", ".."))
data_dir = Path(join(root_dir, "data"))
validation_data_dir = Path(join(testdir, "validation_data"))

class TestShell(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def test_baseline_validation(self):
        """Test against baseline results for a segment"""
        segment_file = data_dir / "shell_blade" / "bar_urc_shell_mesh_segment_2.yaml"

        segment_mesh = ShellSegmentMesh(str(segment_file))

        segment_mesh.generate_mesh_file("test_shell_segment.msh")
        assert filecmp.cmp(
            "test_shell_segment.msh", validation_data_dir / "test_shell_segment.msh"
        )

        # ABD computation
        abd = segment_mesh.compute_ABD()
        abd_concat = np.concatenate(abd)

        expected_abd = np.loadtxt(join(validation_data_dir, "test_shell_abd.txt"))
        assert np.isclose(abd_concat, expected_abd).all()

        # Stiffness computation
        timo_seg_stiffness, eb_seg_stiffness, l_timo_stiffness, r_timo_stiffness = (
            segment_mesh.compute_stiffness(abd)
        )

        # Validate results against baseline
        test_timo_seg_stiffness = np.loadtxt(
            join(validation_data_dir, "test_shell_timo_seg_stiffness.txt")
        )
        test_eb_seg_stiffness = np.loadtxt(
            join(validation_data_dir, "test_shell_eb_seg_stiffness.txt")
        )
        test_l_timo_stiffness = np.loadtxt(
            join(validation_data_dir, "test_shell_l_timo_stiffness.txt")
        )
        test_r_timo_stiffness = np.loadtxt(
            join(validation_data_dir, "test_shell_r_timo_stiffness.txt")
        )

        # print("TROUBLESHOOT TEST")
        # print((timo_seg_stiffness - test_timo_seg_stiffness).max())
        # print((eb_seg_stiffness - test_eb_seg_stiffness).max())
        # print((l_timo_stiffness - test_l_timo_stiffness).max())
        # print((r_timo_stiffness - test_r_timo_stiffness).max())

        # print((timo_seg_stiffness - test_timo_seg_stiffness).min())
        # print((eb_seg_stiffness - test_eb_seg_stiffness).min())
        # print((l_timo_stiffness - test_l_timo_stiffness).min())
        # print((r_timo_stiffness - test_r_timo_stiffness).min())

        assert np.isclose(
            timo_seg_stiffness, test_timo_seg_stiffness, rtol=1e-03, atol=1e-04
        ).all()
        assert np.isclose(
            eb_seg_stiffness, test_eb_seg_stiffness, rtol=1e-03, atol=1e-04
        ).all()
        assert np.isclose(
            l_timo_stiffness, test_l_timo_stiffness, rtol=1e-03, atol=1e-04
        ).all()
        assert np.isclose(
            r_timo_stiffness, test_r_timo_stiffness, rtol=1e-03, atol=1e-04
        ).all()

        # Remove generated files
        os.remove("test_shell_segment.msh")

        print("Baseline validation passed!")
        return


def run_workflow():
    """This function regenerates the test results. Use this if updates to the code have
    changed the expected outputs and these new outputs are what should be tested against.
    """
    mesh_yaml = join(data_dir, "shell_blade", "bar_urc_shell_mesh.yaml")

    opensg.io.generate_segment_shell_mesh_files(
        mesh_yaml, segment_folder=data_dir, segment_list=[2]
    )

    segment_file = data_dir / "bar_urc_shell_mesh_segment_2.yaml"

    segment_mesh = ShellSegmentMesh(str(segment_file))

    segment_mesh.generate_mesh_file(validation_data_dir / "test_shell_segment.msh")

    abd = segment_mesh.compute_ABD()
    abd_concat = np.concatenate(abd)

    np.savetxt(validation_data_dir / "test_shell_abd.txt", abd_concat)

    timo_seg_stiffness, eb_seg_stiffness, l_timo_stiffness, r_timo_stiffness = (
        segment_mesh.compute_stiffness(abd)
    )

    np.savetxt(
        join(validation_data_dir, "test_shell_timo_seg_stiffness.txt"),
        timo_seg_stiffness,
    )
    np.savetxt(
        join(validation_data_dir, "test_shell_eb_seg_stiffness.txt"), eb_seg_stiffness
    )
    np.savetxt(
        join(validation_data_dir, "test_shell_l_timo_stiffness.txt"), l_timo_stiffness
    )
    np.savetxt(
        join(validation_data_dir, "test_shell_r_timo_stiffness.txt"), r_timo_stiffness
    )

    return


if __name__ == "__main__":
    run_workflow()
    unittest.main()
