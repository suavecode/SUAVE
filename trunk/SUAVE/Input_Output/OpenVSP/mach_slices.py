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

def mach_slices(vehicle,mach,angle_of_attack=0.,number_slices = 100):
    
    # Write the vehicle
    write(vehicle,vehicle.tag,write_file=False)
    
    # Calculate the mach angle and adjust for AoA
    mach_angle = np.arcsin(1/mach[0])
    roty = mach_angle - angle_of_attack[0]
    
    # Take the components of the X and Z axis to get the slicing plane
    x_component = np.sin(roty)
    z_component = np.cos(roty)
    
    # Now slice it 
    slice_mesh_id = vsp.ComputePlaneSlice( 0, number_slices, vsp.vec3d(x_component, 0.0, z_component), True)
    
    # Pull out the areas from the slices
    pslice_results = vsp.FindLatestResultsID("Slice")
    slice_areas    = vsp.GetDoubleResults( pslice_results, "Slice_Area" )
    
    X_locs = np.linspace(0,vehicle.total_length,number_slices,)
    
    return X_locs, slice_areas