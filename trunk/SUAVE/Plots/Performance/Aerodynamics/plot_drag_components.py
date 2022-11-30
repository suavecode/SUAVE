## @ingroup Plots-Performance-Aerodynamics
# plot_drag_components.py
# 
# Created:    Nov 2022, E. Botero
# Modified:   

# ----------------------------------------------------------------------
#   Imports
# ---------------------------------------------------------------------- 

from SUAVE.Core import Units

# ---------------------------------------------------------------------- 
#   Drag Components
# ---------------------------------------------------------------------- 

## @ingroup Plots-Performance-Aerodynamics
def plot_drag_components(results):
    """This plots the drag components of the aircraft
    
    Assumptions:
    None
    
    Source:
    None
    
    Inputs:
    results.segments.condtions.aerodynamics.drag_breakdown
          parasite.total
          induced.total
          compressible.total
          miscellaneous.total
          
    Outputs:
    Plots
    
    Properties Used:
    N/A
    """
    for i, segment in enumerate(results.segments.values()):
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        drag_breakdown = segment.conditions.aerodynamics.drag_breakdown
        cdp = drag_breakdown.parasite.total[:,0]
        cdi = drag_breakdown.induced.total[:,0]
        cdc = drag_breakdown.compressible.total[:,0]
        cdm = drag_breakdown.miscellaneous.total[:,0]
        cd  = drag_breakdown.total[:,0]

    return
