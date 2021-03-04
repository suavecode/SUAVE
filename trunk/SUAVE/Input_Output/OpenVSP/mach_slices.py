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

def mach_slices(vehicle,mach,angle_of_attack=[0.],number_slices = 100):
    """This function computes mach slices for a given flight condition
    
    Assumptions:
    N/A

    Source:
    "Two Supersonic Business Aircraft Conceptual Designs, With and Without Sonic Boom Constraint" by
    Aronstein and Schueler

    Inputs:
    vehicle              SUAVE Vehicle       [-]
    mach                 mach number         [-]
    angle_of_attack      angle of attack     [radians]
    number_slices        number of slices    [-]
    

    Outputs:    
    X_locs             location where the slice crosses the X-axis [m]
    slice_areas        cross sectional area                        [m^2]
      

    Properties Used:
    N/A
    """       
    
    # Write the vehicle
    write(vehicle,vehicle.tag,write_file=False)
    
    # Calculate the mach angle and adjust for AoA
    mach_angle = np.arcsin(1/mach[0])
    roty = mach_angle - angle_of_attack[0]
    
    # Take the components of the X and Z axis to get the slicing plane
    x_component = np.cos(roty)
    z_component = np.sin(roty)
    
    # Now slice it 
    vsp.ComputePlaneSlice( 0, number_slices, vsp.vec3d(x_component, 0.0, z_component), True)
    
    # Pull out the areas from the slices
    pslice_results = vsp.FindLatestResultsID("Slice")
    slice_areas    = vsp.GetDoubleResults( pslice_results, "Slice_Area" )
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
        X_locs      = X_locs[0:-1]
        
    return X_locs, slice_areas