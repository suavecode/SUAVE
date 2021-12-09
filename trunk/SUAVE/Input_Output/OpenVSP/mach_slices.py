## @ingroup Input_Output-OpenVSP
# mach_slices.py
# Created:  May 2021, E. Botero
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

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
    """ This method calculates the volume equivalent area for a vehicle for sonic boom analysis. It will write a VSP
        then slice it up depending on the mach number and angle of attack
    
        Assumptions:
        X_locs is the location where lift values are taken on the x-axis
        AE_x is the lift equivalent area
        
        
        Source:
        N/A
        
        Inputs:
        vehicle             [vehicle]
        mach                [-]
        angle_of_attack     [radians]
        number_slices       [int]


        Outputs:
        X_locs              [m]
        slice_areas         [m^2]
        
        Properties Used:
        N/A
    """       
    

    # Write the vehicle
    write(vehicle,vehicle.tag,write_file=False)
    
    X_locs_all       = []
    slice_areas_all = []
    
    for ii in range(len(mach)):
        
        m   = mach[ii] 
        
        if len(angle_of_attack)>1:
            aoa = angle_of_attack[ii]
        else:
            aoa = angle_of_attack[0]
        
    
        # Calculate the mach angle and adjust for AoA
        mach_angle = np.arcsin(1/m)
        roty = (np.pi/2-mach_angle) + aoa
        
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
        
        # A vectorized Output
        X_locs_all.append(X_locs)
        slice_areas_all.append(slice_areas)

    return X_locs_all, slice_areas_all