## @ingroup Plots-Performance-Aerodynamics
# plot_stability_coefficients.py
# 
# Created:    Nov 2022, E. Botero
# Modified:   

# ----------------------------------------------------------------------
#   Imports
# ---------------------------------------------------------------------- 

from SUAVE.Core import Units

# ---------------------------------------------------------------------- 
#   Stability Coefficients
# ---------------------------------------------------------------------- 

## @ingroup Plots-Performance-Aerodynamics
def plot_stability_coefficients(results):
    """This plots the static stability characteristics of an aircraft
    
    Assumptions:
    None
    
    Source:
    None
    
    Inputs:
    results.segments.conditions.stability.
       static
           CM
           Cm_alpha
           static_margin
       aerodynamics.
           angle_of_attack
    Outputs:
    
    Plots
    Properties Used:
    N/A
    """

    for segment in results.segments.values():
        time     = segment.conditions.frames.inertial.time[:,0] / Units.min
        cm       = segment.conditions.stability.static.CM[:,0]
        cm_alpha = segment.conditions.stability.static.Cm_alpha[:,0]
        SM       = segment.conditions.stability.static.static_margin[:,0]
        aoa      = segment.conditions.aerodynamics.angle_of_attack[:,0] / Units.deg


    return
