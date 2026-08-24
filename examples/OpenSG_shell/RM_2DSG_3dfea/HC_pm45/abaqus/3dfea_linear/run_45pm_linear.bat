@echo off
REM run_45pm_linear.bat -- the SAME pm45 L_min deck with nlgeom=NO
REM (geometrically linear), for the apples-to-apples comparison
REM with the linear OpenSG/SwiftComp chain.  Outputs carry the
REM _linear suffix so the nonlinear reference is never overwritten.
set SH=\\roger.ecn.purdue.edu\bagla0\OpenSG-2.0\examples\OpenSG_shell\RM_2DSG_3dfea\HC_pm45\abaqus\3dfea_linear
set JD=C:\Temp\PM45LIN
if exist %JD% rmdir /s /q %JD%
mkdir %JD%
cd /d %JD%
echo === FREE SPACE ===
wmic logicaldisk where "DeviceID='C:'" get FreeSpace /format:list
xcopy "%SH%\45pm_L_min_pm45_linear.inp" %JD%\ /Y
xcopy "%SH%\abq_dump_45pm_linear.py" %JD%\ /Y
call abaqus job=45pm_L_min_pm45_linear cpus=4 interactive
echo ===JOB DONE===
call abaqus python abq_dump_45pm_linear.py
echo ===DUMP DONE===
xcopy %JD%\*.sta "%SH%\" /Y
xcopy %JD%\*.dat "%SH%\" /Y
xcopy %JD%\*.msg "%SH%\" /Y
xcopy %JD%\45pm_S_global_linear.csv "%SH%\" /Y
xcopy %JD%\45pm_S_material_linear.csv "%SH%\" /Y
xcopy %JD%\45pm_U_linear.csv "%SH%\" /Y
echo ===ALL DONE===
