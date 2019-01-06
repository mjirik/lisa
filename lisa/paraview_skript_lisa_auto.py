"""
Script creates pvsm file with colors and video and screenshot.
All is based on recentely created vtk file in ~/lisa_data. All vtk files with the same beginning of file name are
concidered.

Use:
Paraview -> View -> Python Shell -> Run Script
Paraview -> Tools -> Python Shell -> Run Script
"""

import sys
import os.path as op
import glob



files = sorted(glob.glob(op.expanduser("~/lisa_data/*.vtk")), key=op.getmtime)
print(files[0])
print(files[-1])

base, fn = op.split(files[-1])
print(fn)
spl = fn.split("_")
print(spl)


vtk_files = glob.glob(op.expanduser("~/lisa_data/{}_*.vtk".format(spl[0])))
print(vtk_files)
png_file = op.expanduser("~/lisa_data/{}.png".format(spl[0]))
avi_file = op.expanduser("~/lisa_data/{}.avi".format(spl[0]))

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [876, 571]

# Properties modified on renderView1
renderView1.Background = [1.0, 1.0, 1.0]

def look_for_file_containing(vtk_files, patch):

    lfile = None
    for fn in vtk_files:
        if patch in fn:
            if lfile is None:
                lfile = fn
            else:
                # it there are more files matching the patch, the shortest is selected
                if len(lfile) > len(fn):
                    lfile = fn

    if lfile is not None:
        vtk_files.remove(lfile)
    return lfile

# if it is not set, there cannot be focus on this later in the script
a89502994_all_livervtk = None

# get color transfer function/color map for 'node_groups'
node_groupsLUT = GetColorTransferFunction('node_groups')
# node_groupsLUT = GetColorTransferFunction('Solid Color')

# get opacity transfer function/opacity map for 'node_groups'
node_groupsPWF = GetOpacityTransferFunction('node_groups')
# read liver
liver_file = look_for_file_containing(vtk_files, "liver")
if liver_file is not None:
    # create a new 'Legacy VTK Reader'
    a89502994_all_livervtk = LegacyVTKReader(FileNames=[liver_file])
    # a89502994_all_livervtk = LegacyVTKReader(FileNames=['C:\\Users\\miros\\lisa_data\\89502994_all_liver.vtk'])

    # set active source
    SetActiveSource(a89502994_all_livervtk)

    RenameSource('liver', a89502994_all_livervtk)
    # show data in view
    a89502994_all_livervtkDisplay = Show(a89502994_all_livervtk, renderView1)
    # trace defaults for the display properties.
    a89502994_all_livervtkDisplay.Representation = 'Surface'
    a89502994_all_livervtkDisplay.ColorArrayName = ['POINTS', 'node_groups']
    a89502994_all_livervtkDisplay.LookupTable = node_groupsLUT
    a89502994_all_livervtkDisplay.OSPRayScaleArray = 'node_groups'
    a89502994_all_livervtkDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    a89502994_all_livervtkDisplay.SelectOrientationVectors = 'None'
    a89502994_all_livervtkDisplay.ScaleFactor = 18.69596881866455
    a89502994_all_livervtkDisplay.SelectScaleArray = 'node_groups'
    a89502994_all_livervtkDisplay.GlyphType = 'Arrow'
    a89502994_all_livervtkDisplay.GlyphTableIndexArray = 'node_groups'
    a89502994_all_livervtkDisplay.DataAxesGrid = 'GridAxesRepresentation'
    a89502994_all_livervtkDisplay.PolarAxes = 'PolarAxesRepresentation'
    a89502994_all_livervtkDisplay.ScalarOpacityFunction = node_groupsPWF
    a89502994_all_livervtkDisplay.ScalarOpacityUnitDistance = 13.163361790272823
    a89502994_all_livervtkDisplay.GaussianRadius = 9.347984409332275
    a89502994_all_livervtkDisplay.SetScaleArray = ['POINTS', 'node_groups']
    a89502994_all_livervtkDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    a89502994_all_livervtkDisplay.OpacityArray = ['POINTS', 'node_groups']
    a89502994_all_livervtkDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    # node_groupsLUT = GetColorTransferFunction('node_groups')

    # show color bar/color legend
    a89502994_all_livervtkDisplay.SetScalarBarVisibility(renderView1, False)

    # reset view to fit data
    renderView1.ResetCamera()

    # Hide the scalar bar for this color map if no visible data is colored by it.
    HideScalarBarIfNotNeeded(node_groupsLUT, renderView1)

    # change solid color
    a89502994_all_livervtkDisplay.DiffuseColor = [0.0, 0.0, 0.0]

    # set scalar coloring
    # ColorBy(a89502994_all_livervtkDisplay, ('POINTS', 'node_groups'))
    ColorBy(a89502994_all_livervtkDisplay, ('POINTS', 'Solid Color'))

    # rescale color and/or opacity maps used to include current data range
    a89502994_all_livervtkDisplay.RescaleTransferFunctionToDataRange(True, False)

    # show color bar/color legend
    a89502994_all_livervtkDisplay.SetScalarBarVisibility(renderView1, False)

    # Hide the scalar bar for this color map if no visible data is colored by it.
    HideScalarBarIfNotNeeded(node_groupsLUT, renderView1)

    # Properties modified on a89502994_all_livervtkDisplay
    a89502994_all_livervtkDisplay.Opacity = 0.27

    # change representation type
    a89502994_all_livervtkDisplay.SetRepresentationType('Wireframe')

    # change solid color
    a89502994_all_livervtkDisplay.AmbientColor = [0.0, 0.0, 0.0]

    # Properties modified on a89502994_all_livervtkDisplay
    a89502994_all_livervtkDisplay.Opacity = 0.1

porta_file = look_for_file_containing(vtk_files, "porta")
if porta_file is not None:
    # create a new 'Legacy VTK Reader'
    a89502994_all_portavtk = LegacyVTKReader(FileNames=[porta_file])
    # a89502994_all_portavtk = LegacyVTKReader(FileNames=['C:\\Users\\miros\\lisa_data\\89502994_all_porta.vtk'])

    # set active source
    SetActiveSource(a89502994_all_portavtk)

    RenameSource('porta', a89502994_all_portavtk)
    # show data in view
    a89502994_all_portavtkDisplay = Show(a89502994_all_portavtk, renderView1)
    # trace defaults for the display properties.
    a89502994_all_portavtkDisplay.Representation = 'Surface'
    a89502994_all_portavtkDisplay.ColorArrayName = ['POINTS', 'node_groups']
    a89502994_all_portavtkDisplay.LookupTable = node_groupsLUT
    a89502994_all_portavtkDisplay.OSPRayScaleArray = 'node_groups'
    a89502994_all_portavtkDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    a89502994_all_portavtkDisplay.SelectOrientationVectors = 'None'
    a89502994_all_portavtkDisplay.ScaleFactor = 11.08781967163086
    a89502994_all_portavtkDisplay.SelectScaleArray = 'node_groups'
    a89502994_all_portavtkDisplay.GlyphType = 'Arrow'
    a89502994_all_portavtkDisplay.GlyphTableIndexArray = 'node_groups'
    a89502994_all_portavtkDisplay.DataAxesGrid = 'GridAxesRepresentation'
    a89502994_all_portavtkDisplay.PolarAxes = 'PolarAxesRepresentation'
    a89502994_all_portavtkDisplay.ScalarOpacityFunction = node_groupsPWF
    a89502994_all_portavtkDisplay.ScalarOpacityUnitDistance = 19.126375607748372
    a89502994_all_portavtkDisplay.GaussianRadius = 5.54390983581543
    a89502994_all_portavtkDisplay.SetScaleArray = ['POINTS', 'node_groups']
    a89502994_all_portavtkDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    a89502994_all_portavtkDisplay.OpacityArray = ['POINTS', 'node_groups']
    a89502994_all_portavtkDisplay.OpacityTransferFunction = 'PiecewiseFunction'

    # show color bar/color legend
    a89502994_all_portavtkDisplay.SetScalarBarVisibility(renderView1, False)

    # Hide the scalar bar for this color map if no visible data is colored by it.
    HideScalarBarIfNotNeeded(node_groupsLUT, renderView1)

    # change solid color
    a89502994_all_portavtkDisplay.DiffuseColor = [0.0, 0.0, 1.0]
    ColorBy(a89502994_all_portavtkDisplay, ('POINTS', 'Solid Color'))

hv_file = look_for_file_containing(vtk_files, "hepatic_veins")
print("hepatic veins ", hv_file)
if hv_file is not None:
    # create a new 'Legacy VTK Reader'
    a89502994_all_hepatic_veinsvtk = LegacyVTKReader(FileNames=[hv_file])


    # set active source
    SetActiveSource(a89502994_all_hepatic_veinsvtk)
    RenameSource('hepatic_veins', a89502994_all_hepatic_veinsvtk)

    # show data in view
    a89502994_all_hepatic_veinsvtkDisplay = Show(a89502994_all_hepatic_veinsvtk, renderView1)
    # trace defaults for the display properties.
    a89502994_all_hepatic_veinsvtkDisplay.Representation = 'Surface'
    a89502994_all_hepatic_veinsvtkDisplay.ColorArrayName = ['POINTS', 'node_groups']
    a89502994_all_hepatic_veinsvtkDisplay.LookupTable = node_groupsLUT
    a89502994_all_hepatic_veinsvtkDisplay.OSPRayScaleArray = 'node_groups'
    a89502994_all_hepatic_veinsvtkDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    a89502994_all_hepatic_veinsvtkDisplay.SelectOrientationVectors = 'None'
    a89502994_all_hepatic_veinsvtkDisplay.ScaleFactor = 11.108200454711914
    a89502994_all_hepatic_veinsvtkDisplay.SelectScaleArray = 'node_groups'
    a89502994_all_hepatic_veinsvtkDisplay.GlyphType = 'Arrow'
    a89502994_all_hepatic_veinsvtkDisplay.GlyphTableIndexArray = 'node_groups'
    a89502994_all_hepatic_veinsvtkDisplay.DataAxesGrid = 'GridAxesRepresentation'
    a89502994_all_hepatic_veinsvtkDisplay.PolarAxes = 'PolarAxesRepresentation'
    a89502994_all_hepatic_veinsvtkDisplay.ScalarOpacityFunction = node_groupsPWF
    a89502994_all_hepatic_veinsvtkDisplay.ScalarOpacityUnitDistance = 20.819237020536345
    a89502994_all_hepatic_veinsvtkDisplay.GaussianRadius = 5.554100227355957
    a89502994_all_hepatic_veinsvtkDisplay.SetScaleArray = ['POINTS', 'node_groups']
    a89502994_all_hepatic_veinsvtkDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    a89502994_all_hepatic_veinsvtkDisplay.OpacityArray = ['POINTS', 'node_groups']
    a89502994_all_hepatic_veinsvtkDisplay.OpacityTransferFunction = 'PiecewiseFunction'

    # show color bar/color legend
    a89502994_all_hepatic_veinsvtkDisplay.SetScalarBarVisibility(renderView1, False)

    # Hide the scalar bar for this color map if no visible data is colored by it.
    HideScalarBarIfNotNeeded(node_groupsLUT, renderView1)

    # change solid color
    a89502994_all_hepatic_veinsvtkDisplay.DiffuseColor = [1.0, 0.0, 0.0]
    ColorBy(a89502994_all_hepatic_veinsvtkDisplay, ('POINTS', 'Solid Color'))

print(vtk_files)
tumor_file = look_for_file_containing(vtk_files, "tumor")
if tumor_file is not None:
    # create a new 'Legacy VTK Reader'
    a89502994_all_tumorvtk = LegacyVTKReader(FileNames=[tumor_file])

    # set active source
    SetActiveSource(a89502994_all_tumorvtk)
    RenameSource('tumor', a89502994_all_tumorvtk)

    # show data in view
    a89502994_all_tumorvtkDisplay = Show(a89502994_all_tumorvtk, renderView1)
    # trace defaults for the display properties.
    a89502994_all_tumorvtkDisplay.Representation = 'Surface'
    a89502994_all_tumorvtkDisplay.ColorArrayName = ['POINTS', 'node_groups']
    a89502994_all_tumorvtkDisplay.LookupTable = node_groupsLUT
    a89502994_all_tumorvtkDisplay.OSPRayScaleArray = 'node_groups'
    a89502994_all_tumorvtkDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    a89502994_all_tumorvtkDisplay.SelectOrientationVectors = 'None'
    a89502994_all_tumorvtkDisplay.ScaleFactor = 3.433440399169922
    a89502994_all_tumorvtkDisplay.SelectScaleArray = 'node_groups'
    a89502994_all_tumorvtkDisplay.GlyphType = 'Arrow'
    a89502994_all_tumorvtkDisplay.GlyphTableIndexArray = 'node_groups'
    a89502994_all_tumorvtkDisplay.DataAxesGrid = 'GridAxesRepresentation'
    a89502994_all_tumorvtkDisplay.PolarAxes = 'PolarAxesRepresentation'
    a89502994_all_tumorvtkDisplay.ScalarOpacityFunction = node_groupsPWF
    a89502994_all_tumorvtkDisplay.ScalarOpacityUnitDistance = 7.888198796791789
    a89502994_all_tumorvtkDisplay.GaussianRadius = 1.716720199584961
    a89502994_all_tumorvtkDisplay.SetScaleArray = ['POINTS', 'node_groups']
    a89502994_all_tumorvtkDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    a89502994_all_tumorvtkDisplay.OpacityArray = ['POINTS', 'node_groups']
    a89502994_all_tumorvtkDisplay.OpacityTransferFunction = 'PiecewiseFunction'

    # show color bar/color legend
    # a89502994_all_tumorvtkDisplay.SetScalarBarVisibility(renderView1, True)
    a89502994_all_tumorvtkDisplay.SetScalarBarVisibility(renderView1, False)

    # Hide the scalar bar for this color map if no visible data is colored by it.
    HideScalarBarIfNotNeeded(node_groupsLUT, renderView1)

    # change solid color
    a89502994_all_tumorvtkDisplay.DiffuseColor = [0.6666666666666666, 0.3333333333333333, 0.0]
    ColorBy(a89502994_all_tumorvtkDisplay, ('POINTS', 'Solid Color'))

lungs_file = look_for_file_containing(vtk_files, "lungs")
print("lungs", lungs_file)
if lungs_file is not None:
    # create a new 'Legacy VTK Reader'
    a89502994_all_lungs_vtk = LegacyVTKReader(FileNames=[lungs_file])


    # set active source
    SetActiveSource(a89502994_all_lungs_vtk)
    RenameSource('lungs', a89502994_all_lungs_vtk)

    # show data in view
    a89502994_all_lungs_vtkDisplay = Show(a89502994_all_lungs_vtk, renderView1)
    # # trace defaults for the display properties.
    # a89502994_all_lungs_vtkDisplay.Representation = 'Surface'
    # a89502994_all_lungs_vtkDisplay.ColorArrayName = ['POINTS', 'node_groups']
    # a89502994_all_lungs_vtkDisplay.LookupTable = node_groupsLUT
    # a89502994_all_lungs_vtkDisplay.OSPRayScaleArray = 'node_groups'
    # a89502994_all_lungs_vtkDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    # a89502994_all_lungs_vtkDisplay.SelectOrientationVectors = 'None'
    # a89502994_all_lungs_vtkDisplay.ScaleFactor = 11.108200454711914
    # a89502994_all_lungs_vtkDisplay.SelectScaleArray = 'node_groups'
    # a89502994_all_lungs_vtkDisplay.GlyphType = 'Arrow'
    # a89502994_all_lungs_vtkDisplay.GlyphTableIndexArray = 'node_groups'
    # a89502994_all_lungs_vtkDisplay.DataAxesGrid = 'GridAxesRepresentation'
    # a89502994_all_lungs_vtkDisplay.PolarAxes = 'PolarAxesRepresentation'
    # a89502994_all_lungs_vtkDisplay.ScalarOpacityFunction = node_groupsPWF
    # a89502994_all_lungs_vtkDisplay.ScalarOpacityUnitDistance = 20.819237020536345
    # a89502994_all_lungs_vtkDisplay.GaussianRadius = 5.554100227355957
    # a89502994_all_lungs_vtkDisplay.SetScaleArray = ['POINTS', 'node_groups']
    # a89502994_all_lungs_vtkDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    # a89502994_all_lungs_vtkDisplay.OpacityArray = ['POINTS', 'node_groups']
    # a89502994_all_lungs_vtkDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    #
    # show color bar/color legend
    a89502994_all_lungs_vtkDisplay.SetScalarBarVisibility(renderView1, False)

    # Properties modified on a89502994_all_livervtkDisplay
    a89502994_all_lungs_vtkDisplay.Opacity = 0.05

    # Hide the scalar bar for this color map if no visible data is colored by it.
    HideScalarBarIfNotNeeded(node_groupsLUT, renderView1)

    # change solid color
    a89502994_all_lungs_vtkDisplay.DiffuseColor = [0.1, 0.1, 0.2]
    ColorBy(a89502994_all_lungs_vtkDisplay, ('POINTS', 'Solid Color'))


bones_file = look_for_file_containing(vtk_files, "bones")
print("bones", bones_file)
if bones_file is not None:
    # create a new 'Legacy VTK Reader'
    a89502994_all_bones_vtk = LegacyVTKReader(FileNames=[bones_file])


    # set active source
    SetActiveSource(a89502994_all_bones_vtk)
    RenameSource('bones', a89502994_all_bones_vtk)

    # show data in view
    a89502994_all_bones_vtkDisplay = Show(a89502994_all_bones_vtk, renderView1)
    # # trace defaults for the display properties.
    # a89502994_all_bones_vtkDisplay.Representation = 'Surface'
    # a89502994_all_bones_vtkDisplay.ColorArrayName = ['POINTS', 'node_groups']
    # a89502994_all_bones_vtkDisplay.LookupTable = node_groupsLUT
    # a89502994_all_bones_vtkDisplay.OSPRayScaleArray = 'node_groups'
    # a89502994_all_bones_vtkDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    # a89502994_all_bones_vtkDisplay.SelectOrientationVectors = 'None'
    # a89502994_all_bones_vtkDisplay.ScaleFactor = 11.108200454711914
    # a89502994_all_bones_vtkDisplay.SelectScaleArray = 'node_groups'
    # a89502994_all_bones_vtkDisplay.GlyphType = 'Arrow'
    # a89502994_all_bones_vtkDisplay.GlyphTableIndexArray = 'node_groups'
    # a89502994_all_bones_vtkDisplay.DataAxesGrid = 'GridAxesRepresentation'
    # a89502994_all_bones_vtkDisplay.PolarAxes = 'PolarAxesRepresentation'
    # a89502994_all_bones_vtkDisplay.ScalarOpacityFunction = node_groupsPWF
    # a89502994_all_bones_vtkDisplay.ScalarOpacityUnitDistance = 20.819237020536345
    # a89502994_all_bones_vtkDisplay.GaussianRadius = 5.554100227355957
    # a89502994_all_bones_vtkDisplay.SetScaleArray = ['POINTS', 'node_groups']
    # a89502994_all_bones_vtkDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    # a89502994_all_bones_vtkDisplay.OpacityArray = ['POINTS', 'node_groups']
    # a89502994_all_bones_vtkDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    #
    # show color bar/color legend
    a89502994_all_bones_vtkDisplay.SetScalarBarVisibility(renderView1, False)

    # Properties modified on a89502994_all_livervtkDisplay
    a89502994_all_bones_vtkDisplay.Opacity = 0.05

    # Hide the scalar bar for this color map if no visible data is colored by it.
    HideScalarBarIfNotNeeded(node_groupsLUT, renderView1)

    # change solid color
    a89502994_all_bones_vtkDisplay.DiffuseColor = [0.2, 0.2, 0.2]
    ColorBy(a89502994_all_bones_vtkDisplay, ('POINTS', 'Solid Color'))

# see all other

for vtk_file in vtk_files:
    rdr = LegacyVTKReader(FileNames=[vtk_file])
    # a89502994_all_hepatic_veinsvtkDisplay.SetScalarBarVisibility(renderView1, False)



if a89502994_all_livervtk is not None:
    print("nastavujeme active na liver")
    # set active source
    SetActiveSource(a89502994_all_livervtk)

# reset view to fit data
renderView1.ResetCamera()

# get animation scene
animationScene1 = GetAnimationScene()

# Properties modified on animationScene1
animationScene1.NumberOfFrames = 150

# get camera animation track for the view
cameraAnimationCue1 = GetCameraTrack(view=renderView1)

# create keyframes for this animation track

# create a key frame
keyFrame6526 = CameraKeyFrame()
keyFrame6526.Position = [121.41339302062988, -445.04176535120837, 108.47465419769287]
keyFrame6526.FocalPoint = [121.41339302062988, 118.78985023498535, 108.47465419769287]
keyFrame6526.ViewUp = [-1.0, 0.0, 0.0]
keyFrame6526.ParallelScale = 145.9303603446302
keyFrame6526.PositionPathPoints = [121.413, -445.042, 108.475, 121.413, -236.04097472641195, 546.6547617402068, 121.413, 236.01726445715883, 659.9859180169432, 121.413, 621.1679905461758, 364.4493714491482, 121.413, 633.876162473543, -120.85613493871458, 121.413, 264.72045983824455, -436.14489048821804, 121.413, -212.6221343705697, -347.6746699724155]
keyFrame6526.FocalPathPoints = [121.413, 118.79, 108.475]
keyFrame6526.ClosedPositionPath = 1

# create a key frame
keyFrame6527 = CameraKeyFrame()
keyFrame6527.KeyTime = 1.0
keyFrame6527.Position = [121.41339302062988, -445.04176535120837, 108.47465419769287]
keyFrame6527.FocalPoint = [121.41339302062988, 118.78985023498535, 108.47465419769287]
keyFrame6527.ViewUp = [-1.0, 0.0, 0.0]
keyFrame6527.ParallelScale = 145.9303603446302

# initialize the animation track
cameraAnimationCue1.Mode = 'Path-based'
cameraAnimationCue1.KeyFrames = [keyFrame6526, keyFrame6527]

# current camera placement for renderView1
renderView1.CameraPosition = [121.41339302062988, -445.04176535120837, 108.47465419769287]
renderView1.CameraFocalPoint = [121.41339302062988, 118.78985023498535, 108.47465419769287]
renderView1.CameraViewUp = [-1.0, 0.0, 0.0]
renderView1.CameraParallelScale = 145.9303603446302

# save screenshot
# SaveScreenshot('C:\Users\miros\lisa_data\89502994_auto.png', renderView1, ImageResolution=[1752, 1142])
SaveScreenshot(png_file, renderView1, ImageResolution=[1752, 1142])

# current camera placement for renderView1
renderView1.CameraPosition = [121.41339302062988, -445.04176535120837, 108.47465419769287]
renderView1.CameraFocalPoint = [121.41339302062988, 118.78985023498535, 108.47465419769287]
renderView1.CameraViewUp = [-1.0, 0.0, 0.0]
renderView1.CameraParallelScale = 145.9303603446302

# save animation
# SaveAnimation('C:\Users\miros\lisa_data\89502994.avi', renderView1, ImageResolution=[876, 568],
SaveAnimation(avi_file, renderView1, ImageResolution=[876, 568],
    FrameRate=15,
    FrameWindow=[0, 149])

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.CameraPosition = [121.413, -445.042, 108.475]
renderView1.CameraFocalPoint = [121.413, 118.79, 108.475]
renderView1.CameraViewUp = [-1.0, 0.0, 0.0]
renderView1.CameraParallelScale = 145.9303603446302

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).