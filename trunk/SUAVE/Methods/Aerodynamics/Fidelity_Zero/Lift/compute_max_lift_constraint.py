## @ingroup Methods-Constraint_Analysis
# Oswald_efficiency.py
# 
# Created:  Nov 2021, S. Karpuk
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE Imports
import SUAVE
from SUAVE.Core import Units

# ------------------------------------------------------------------------------------
#  Compute maximum lift coefficient for the constraint analysis
# ------------------------------------------------------------------------------------

## @ingroup Methods-Constraint_Analysis
def compute_max_lift_constraint(constraint_analysis):
    """Estimates the wing maximum lift coefficient for the constraint analysis using the least possible number of variables

        Assumptions:
        None

        Source:
            D. Scholz, "Aircraft Design lecture notes", https://www.fzt.haw-hamburg.de/pers/Scholz/HOOU/AircraftDesign_5_PreliminarySizing.pdf

        Inputs:
            self.geometry.sweep_quarter_chord       [degrees]

        Outputs:
            CLmax       [Unitless]

        Properties Used:

    """  

    # Unpack inputs
    highlift_type   = constraint_analysis.high_lift_configuration_type
    sweep           = constraint_analysis.sweep_quarter_chord / Units.degrees

    if highlift_type == None:                                           # No Flaps
        return -0.0002602 * sweep**2 -0.0008614 * sweep + 1.51    
            
    elif highlift_type == 'plain':                                      # Plain 
        return -0.0002823 * sweep**2 -0.000141 * sweep + 1.81  
            
    elif highlift_type == 'single-slotted':                             # Single-Slotted 
        return -0.0002599 * sweep**2 -0.002727 * sweep + 2.205      
      
    elif highlift_type == 'single-slotted Fowler':                      # Fowler
        return -0.0002830 * sweep**2 -0.003897 * sweep + 2.501  
      
    elif highlift_type == 'double-slotted fixed vane':                  # Double-Slotted            
        return -0.0002574 * sweep**2 -0.007129 * sweep + 2.735  
            
    elif highlift_type == 'double-slotted Fowler':                      # Double-slotted Fowler with Slats
        return -0.0002953 * sweep**2 -0.006719 * sweep + 3.014 
            
    elif highlift_type == 'triple-slotted Fowler':                      # Triple-slotted Fowler with Slats
        return -0.0003137 * sweep**2 -0.008903 * sweep + 3.416 
              
    else:
        highliftmsg = "High-lift device type must be specified in the design dictionary."
        raise ValueError(highliftmsg)




    
