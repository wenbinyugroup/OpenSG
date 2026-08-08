"""Render OpenSG dehomogenization VTK results to PNG with ParaView.

Run with pvpython (ParaView ships its own Python), NOT the OpenSG env:
    "C:/Program Files/ParaView-5.12.0-Windows-Python3.10-msvc2017-AMD64/bin/pvpython.exe" render_pv.py

The server has no OpenGL, so these renders are produced locally against the
mounted repository. Every contour is drawn from the exploded Gauss-point mesh
with flat (cell) shading, so no value is smoothed across an element boundary.

In:  VTK files listed in CASES, each an unstructured grid carrying cell or
     point scalars named in the case tuple
Out: one PNG per case under docs/_static
"""
import os

from paraview.simple import (OpenDataFile, GetActiveViewOrCreate, Show, Hide,
                             ColorBy, GetColorTransferFunction,
                             GetScalarBar, SaveScreenshot, Delete)

REPO = "Y:/OpenSG-2.0"
OUT = REPO + "/docs/_static"

# (vtk path, array name, association, title, output png, view direction)
CASES = [
    (REPO + "/examples/OpenSG_shell/2_get_beam_dehom_from_shell_cross_section/"
     "iea_s10_dehom_shell.vtk", "S11", "CELLS",
     "sigma_11 (MPa)", "pv_shell_dehom_s11.png", "xy"),
    (REPO + "/examples/OpenSG_shell/2_get_beam_dehom_from_shell_cross_section/"
     "iea_s10_dehom_shell.vtk", "S12", "CELLS",
     "sigma_12 (MPa)", "pv_shell_dehom_s12.png", "xy"),
    (REPO + "/examples/OpenSG-solid/4_get_plate_dehom_from_2DSG/"
     "RHC_SW_2UC_45_dehom.vtk", "Sig_xx", "POINTS",
     "sigma_xx", "pv_plate_dehom_sxx.png", "xy"),
    (REPO + "/examples/OpenSG-solid/8_get_beam_dehom_from_SG/"
     "RHC_SW_2UC_45_dehom.vtk", "Sig_xx", "POINTS",
     "sigma_xx", "pv_beam_dehom_sxx.png", "xy"),
    (REPO + "/examples/OpenSG-solid/2_get_plate_stress_from_1DSG/"
     "ex5_shell_S8R_field_ff_gauss.vtk", "S11", "POINTS",
     "sigma_11", "pv_plate_1dsg_s11.png", "xy"),
    (REPO + "/examples/OpenSG-solid/2_get_plate_stress_from_1DSG/"
     "ex5_shell_S8R_field_ff_gauss.vtk", "S13", "POINTS",
     "sigma_13", "pv_plate_1dsg_s13.png", "xy"),
]


def render(path, array, assoc, title, png, view):
    """Render one scalar field of one VTK file to a PNG.

    In:  path str VTK file; array str scalar name; assoc "CELLS"|"POINTS";
         title str colour-bar title; png str output file name; view str
         camera axis pair
    Out: writes OUT/png; returns the data range (lo, hi) or None if the
         array is absent
    """
    src = OpenDataFile(path)
    src.UpdatePipeline()
    names = ([src.CellData.GetArray(i).Name
              for i in range(src.CellData.GetNumberOfArrays())]
             if assoc == "CELLS" else
             [src.PointData.GetArray(i).Name
              for i in range(src.PointData.GetNumberOfArrays())])
    if array not in names:
        print("  SKIP %s: no %s array (have %s)" % (os.path.basename(path),
                                                    array, names))
        Delete(src)
        return None
    arr = (src.CellData[array] if assoc == "CELLS" else src.PointData[array])
    lo, hi = arr.GetRange()

    view_obj = GetActiveViewOrCreate("RenderView")
    disp = Show(src, view_obj)
    disp.SetRepresentationType("Surface")
    ColorBy(disp, (assoc[:-1].capitalize() + "s"
                   if False else ("CELLS" if assoc == "CELLS" else "POINTS"),
                   array))
    lut = GetColorTransferFunction(array)
    # rainbow is the project default: its mid-range green keeps lightly
    # loaded regions (the shear webs) visible, where a diverging map fades
    # them into the white background
    lut.ApplyPreset("Rainbow Uniform", True)
    m = max(abs(lo), abs(hi))
    lut.RescaleTransferFunction(-m, m)
    lut.NanColor = [0.5, 0.5, 0.5]

    disp.SetScalarBarVisibility(view_obj, True)
    bar = GetScalarBar(lut, view_obj)
    bar.Title = title
    bar.ComponentTitle = ""
    bar.TitleFontSize = 20
    bar.LabelFontSize = 16
    bar.TitleColor = [0, 0, 0]
    bar.LabelColor = [0, 0, 0]
    bar.Orientation = "Horizontal"
    bar.WindowLocation = "Any Location"
    bar.Position = [0.22, 0.03]
    bar.ScalarBarLength = 0.56
    bar.ScalarBarThickness = 14
    bar.AutomaticLabelFormat = 0
    bar.LabelFormat = "%-#6.3g"
    bar.RangeLabelFormat = "%-#6.3g"

    view_obj.UseColorPaletteForBackground = 0
    view_obj.Background = [1, 1, 1]
    view_obj.OrientationAxesVisibility = 0
    view_obj.InteractionMode = "2D"

    # size the canvas to the model's own aspect ratio and fit the camera to
    # its bounds, so the figure carries no dead margin
    x0, x1, y0, y1, z0, z1 = src.GetDataInformation().GetBounds()
    w, h = max(x1 - x0, 1e-12), max(y1 - y0, 1e-12)
    aspect = w / h
    base = 1500
    if aspect >= 1.0:
        vw, vh = base, max(int(base / aspect), 520)
    else:
        vh, vw = base, max(int(base * aspect), 520)
    vh += 130                              # room for the horizontal colour bar
    view_obj.ViewSize = [vw, vh]
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    view_obj.CameraPosition = [cx, cy, 1.0]
    view_obj.CameraFocalPoint = [cx, cy, 0.0]
    view_obj.CameraViewUp = [0, 1, 0]
    view_obj.CameraParallelProjection = 1
    view_obj.CameraParallelScale = 0.62 * max(h, w / max(vw / float(vh), 1e-12))

    SaveScreenshot(os.path.join(OUT, png), view_obj,
                   ImageResolution=[vw, vh], TransparentBackground=0)
    print("  wrote %-32s %s range [%.4g, %.4g]" % (png, array, lo, hi))
    Hide(src, view_obj)
    Delete(src)
    return lo, hi


for path, array, assoc, title, png, view in CASES:
    if not os.path.exists(path):
        print("  MISSING %s" % path)
        continue
    render(path, array, assoc, title, png, view)
print("done")
