# section_properties.py
# 
# Created:  Feb 2016, E. Botero
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Units, Data

# ----------------------------------------------------------------------
#  Section Properties
# ----------------------------------------------------------------------

def section_properties(state,settings,geometry):
    """This model is based on the NASA TR: "Models of Lift and Drag Coefficients of Stalled and Unstalled Airfoils in
     Wind Turbines and Wind Tunnels" by D. A. Spera
    
    From RE and t/c, get Clmax, CD0,  ACD1', ACL1'"""
    
    # Unpack
    wing = geometry
    re   = state.conditions.freestream.reynolds_number
    mac  = wing.chords.mean_aerodynamic
    tc   = wing.thickness_to_chord
    A0   = settings.section_zero_lift_angle_of_attack
    S1p  = settings.section_lift_curve_slope
    
    # RE dimensionless
    RE = re*mac
    
    # Calculate 2-D CLmax
    # From 241 A/B notes
    Cl_max_ref = -0.0009*tc**3 + 0.0217*tc**2 - 0.0442*tc + 0.7005
    Re_ref     = 9.*10**6
    #CL1maxp = Cl_max_ref * ( RE / Re_ref ) **0.1
    CL1maxp = 1.5 * np.ones_like(state.conditions.freestream.altitude)
    
    # Estimate the ACL1'
    ACLp = A0 + CL1maxp/S1p + 3. * Units.deg

    # Calculate 2-D Cd0  
    # First calculate CF, from AA 241 A/B Notes
    CF  = 0.455/(np.log(RE)**2.58)
    
    # Find k, from AA 241 A/B Notes
    beta2 = 1
    C     = 1.1
    k1    = 2.*C*tc
    k2    = C*C*(1+5)**tc*tc/2.;
    k     = 1 + k1 + k2;
    
    # Cd0
    Cd0 = k*CF
    
    # Estimate the CD1max'
    # I have no idea
    CD1maxp = 10.*Cd0
    
    # Estimate the ACD1'
    ACD1p = ACLp
    
    # Pack outputs
    wing.section = Data()
    wing.section.maximum_coefficient_lift                 = CL1maxp
    wing.section.zero_lift_drag_coefficient               = Cd0
    wing.section.angle_attack_max_prestall_lift           = ACLp
    wing.section.pre_stall_maximum_drag_coefficient       = CD1maxp
    wing.section.pre_stall_maximum_drag_coefficient_angle = ACD1p 
    
    return RE, CL1maxp, Cd0, ACLp, CD1maxp