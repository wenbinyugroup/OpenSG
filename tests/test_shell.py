import unittest
import numpy as np
from opensg.mesh.segment import ShellSegmentMesh
import filecmp
import os
from tests import data_dir, validation_data_dir

class TestShell(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def test_baseline_validation(self):
        """Test against baseline results for a segment"""
        segment_file = data_dir / "Shell_3D_Taper" / "BAR_URC_numEl_52_segment_2.yaml"

        segment_mesh = ShellSegmentMesh(str(segment_file))

        segment_mesh.generate_mesh_file("test_shell_segment.msh")
        assert filecmp.cmp(
            "test_shell_segment.msh", validation_data_dir / "test_shell_segment.msh"
        )

        # ABD computation
        abd, _ = segment_mesh.compute_ABD()
        abd_concat = np.concatenate(abd)

        expected_abd = np.loadtxt(validation_data_dir / "test_shell_abd.txt")
        assert np.isclose(abd_concat, expected_abd).all()

        # Stiffness computation
        timo_seg_stiffness, l_timo_stiffness, r_timo_stiffness = (
            segment_mesh.compute_stiffness(abd, boun=False)
        )

        # Validate results against baseline
        test_timo_seg_stiffness = np.loadtxt(
            validation_data_dir / "test_shell_timo_seg_stiffness.txt"
        )
        test_l_timo_stiffness = np.loadtxt(
            validation_data_dir / "test_shell_l_timo_stiffness.txt"
        )
        test_r_timo_stiffness = np.loadtxt(
            validation_data_dir / "test_shell_r_timo_stiffness.txt"
        )

        print("TROUBLESHOOT TEST")
        print((timo_seg_stiffness - test_timo_seg_stiffness).max())
        print((l_timo_stiffness - test_l_timo_stiffness).max())
        print((r_timo_stiffness - test_r_timo_stiffness).max())

        print((timo_seg_stiffness - test_timo_seg_stiffness).min())
        print((l_timo_stiffness - test_l_timo_stiffness).min())
        print((r_timo_stiffness - test_r_timo_stiffness).min())

        assert np.isclose(
            timo_seg_stiffness, test_timo_seg_stiffness, rtol=1e-03, atol=1e-04
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
    segment_file = data_dir / "Shell_3D_Taper" / "BAR_URC_numEl_52_segment_2.yaml"

    segment_mesh = ShellSegmentMesh(str(segment_file))

    segment_mesh.generate_mesh_file(validation_data_dir / "test_shell_segment.msh")

    abd, _ = segment_mesh.compute_ABD()
    abd_concat = np.concatenate(abd)

    np.savetxt(validation_data_dir / "test_shell_abd.txt", abd_concat)

    timo_seg_stiffness, l_timo_stiffness, r_timo_stiffness = (
        segment_mesh.compute_stiffness(abd, boun=False)
    )

    np.savetxt(validation_data_dir / "test_shell_timo_seg_stiffness.txt", timo_seg_stiffness)
    np.savetxt(validation_data_dir / "test_shell_l_timo_stiffness.txt", l_timo_stiffness)
    np.savetxt(validation_data_dir / "test_shell_r_timo_stiffness.txt", r_timo_stiffness)

    return


if __name__ == "__main__":
    run_workflow()
    # unittest.main()
