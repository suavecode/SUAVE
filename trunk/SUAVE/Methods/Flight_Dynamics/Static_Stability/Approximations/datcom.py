# datcom.py
#
# Created: Feb 2014, Tim Momose
# IN PROGRESS

# NOTES/QUESTIONS
# - May need to bring in Roskam Figure 8.47 data (Airplane Design Part VI) for supersonic
#    - IS THIS ACTUALLY VALID FOR SUPERSONIC? ROSKAM FRAMES IT AS A SUBSONIC METHOD
# - For now, this uses an assumed 2D lift coefficient slope of 6.13rad^-1, from Roskam.
#   Should possibly later change it to take a Cl_alpha input based on VPM, etc.
# - Possibly add an effective aspect ratio variable to the wing object for winglet effects, etc.

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
import numpy as np
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.convert_sweep import convert_sweep
from SUAVE.Attributes import Units as Units
from SUAVE.Structure import (
    Data, Container, Data_Exception, Data_Warning,
)

# ----------------------------------------------------------------------
#  Method
# ----------------------------------------------------------------------

def datcom(wing,mach):
    """ cL_alpha = SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.datcom(wing,mach)
        This method uses the DATCOM formula to compute dCL/dalpha without 
        correlations for downwash of lifting surfaces further ahead on the 
        aircraft or upwash resulting from the position of the wing on the body.
        
        CAUTION: The method presented here is applicable for subsonic speeds.
        May be inaccurate for transonic or supersonic flight. A correction factor
        for supersonic flight is included, but may not be completely accurate.
        
        Inputs:
            wing - a data dictionary with the fields:
                effective_apsect_ratio - wing aspect ratio [dimensionless]. If 
                this variable is not inlcuded in the input, the method will look
                for a variable named 'aspect_ratio'.
                sweep_le - wing leading-edge sweep angle [radians]
                taper - wing taper ratio [dimensionless]
            mach - flight Mach number [dimensionless]
    
        Outputs:
            cL_alpha - The derivative of 3D lift coefficient with respect to AoA
                
        Assumptions:
            -Mach number should not be transonic
    """         
    
    #Unpack inputs
    try:
        ar = wing.effective_aspect_ratio
    except AttributeError:   
        ar = wing.aspect_ratio
    sweep  = wing.sweep_le
    
    M      = mach
    
    #Compute relevent parameters
    if M < 1:
        beta = np.sqrt(1.0-M**2.0)
    else:
        beta = np.sqrt(M**2.0-1.0)
        
    half_chord_sweep = convert_sweep(wing,0.0,0.5)  #Assumes original sweep 
                                                      #value is at leading edge
    
    #Compute k correction factor for Mach number    
    #First, compute corrected 2D section lift curve slope (C_la) for the given Mach number
    cla = 6.13          #Section C_la at M = 0; Roskam Airplane Design Part VI, Table 8.1
    if M < 1:
        cla_M = cla/beta
    else:
        cla_M = 4.0/beta
    
    k = cla_M/(2.0*np.pi/beta)
    
    #Compute aerodynamic surface 3D lift curve slope using the DATCOM formula
    cL_alpha = 2.0*np.pi*ar/(2.0+((ar**2.0*beta**2.0/k**2.0)*(1.0+(np.tan(half_chord_sweep))**2.0/beta**2.0)+4.0)**0.5)
    
    return cL_alpha
