## @ingroup Plots-Performance-Aerodynamics
# plot_aerodynamic_forces.py
# 
# Created:    Nov 2022, E. Botero
# Modified:   

# ----------------------------------------------------------------------
#   Imports
# ---------------------------------------------------------------------- 

from SUAVE.Core import Units

# ---------------------------------------------------------------------- 
#   Aerodynamic Forces
# ---------------------------------------------------------------------- 

## @ingroup Plots-Performance-Aerodynamics
def plot_aerodynamic_forces(results):
    """This plots the aerodynamic forces
    
    Assumptions:
    None
    
    Source:
    None
    
    Inputs:
    results.segments.condtions.frames
         body.thrust_force_vector
         wind.lift_force_vector
         wind.drag_force_vector
         
    Outputs:
    Plots
    
    Properties Used:
    N/A
    """

    for segment in results.segments.values():
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0]
        Lift   = -segment.conditions.frames.wind.lift_force_vector[:,2]
        Drag   = -segment.conditions.frames.wind.drag_force_vector[:,0]
        eta    = segment.conditions.propulsion.throttle[:,0]


    return
