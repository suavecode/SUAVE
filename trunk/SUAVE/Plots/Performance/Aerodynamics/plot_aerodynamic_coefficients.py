## @ingroup Plots-Performance-Aerodynamics
# plot_aerodynamic_coefficients.py
# 
# Created:    Nov 2022, E. Botero
# Modified:   

# ----------------------------------------------------------------------
#   Imports
# ---------------------------------------------------------------------- 

from SUAVE.Core import Units

# ---------------------------------------------------------------------- 
#   Aerodynamic Coefficients
# ---------------------------------------------------------------------- 

## @ingroup Plots-Performance-Aerodynamics
def plot_aerodynamic_coefficients(results):
    """This plots the aerodynamic coefficients
    
    Assumptions:
    None
    
    Source:
    None
    
    Inputs:
    results.segments.condtions.aerodynamics.
        lift_coefficient
        drag_coefficient
        angle_of_attack
        
    Outputs:
    Plots
    
    Properties Used:
    N/A
    """

    # TODO: Write Function
    for segment in results.segments.values():
        time = segment.conditions.frames.inertial.time[:,0] / Units.min
        cl   = segment.conditions.aerodynamics.lift_coefficient[:,0,None]
        cd   = segment.conditions.aerodynamics.drag_coefficient[:,0,None]
        aoa  = segment.conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        l_d  = cl/cd    

    return
