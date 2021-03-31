## @ingroup Input_Output-OpenVSP
# mach_slices.py

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units, Data
from . import write

try:
    import vsp as vsp
except ImportError:
    # This allows SUAVE to build without OpenVSP
    pass
import numpy as np

# ----------------------------------------------------------------------
#  Mach Slices
# ----------------------------------------------------------------------

def mach_slices(vehicle,mach,angle_of_attack=[0.],number_slices = 99):
    
    vehicle.tag = 'slice'
    
    # Write the vehicle
    write(vehicle,vehicle.tag,write_file=False)
    
    # Calculate the mach angle and adjust for AoA
    mach[0] = 1.
    mach_angle = np.arcsin(1/mach[0])
    roty = (np.pi/2-mach_angle) + angle_of_attack[0]
    
    # Take the components of the X and Z axis to get the slicing plane
    x_component = np.cos(roty)
    z_component = np.sin(roty)
    
    # Now slice it 
    vsp.ComputePlaneSlice( 0, number_slices, vsp.vec3d(x_component[0], 0.0, z_component[0]), True)
    
    # Pull out the areas from the slices
    pslice_results = vsp.FindLatestResultsID("Slice")
    slice_areas    = vsp.GetDoubleResults( pslice_results, "Slice_Area" ) * np.cos(roty)
    vec3d          = vsp.GetVec3dResults(pslice_results, "Slice_Area_Center")
    
    X = []
    Z = []
    
    for v in vec3d:
        X.append(v.x())
        Z.append(v.z())
    
    X = np.array(X)
    Z = np.array(Z)
        
    X_locs = X + Z*np.tan(mach_angle)
    
    if slice_areas[-1]==0.:
        slice_areas = slice_areas[0:-1]
        X_locs      = X_locs[:-1]
        
    # Turn them into arrays
    X_locs      = np.array(X_locs)
    slice_areas = np.array(slice_areas)
           
    return X_locs, slice_areas