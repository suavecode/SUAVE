## @ingroup Methods-Constraint_Analysis
# compute_landing_constraint.py
#
# Created:  Nov 2021, S. Karpuk, 
# Modified: 



# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE Imports
import SUAVE    
from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift.compute_max_lift_constraint           import compute_max_lift_constraint

# package imports
import numpy as np

# ----------------------------------------------------------------------
#  Compute landing constraint
# ----------------------------------------------------------------------

## @ingroup Methods-Constraint_Analysis
def compute_landing_constraint(ca,vehicle):
    
    """Calculate the landing wing loading 

        Assumptions:
        None

        Source:
            L. Loftin,Subsonic Aircraft: Evolution and the Matching of Size to Performance, NASA Ref-erence Publication 1060, August 1980

        Inputs:
            self.aerodynamics.cl_max_landing    [Unitless]
                 landing.ground_roll            [m]
                         runway_elevation       [m]
                         approach_speed_factor  [Unitless]
                         delta_ISA              [K]

        Outputs:
            constraints.T_W                     [Unitless]
                        W_S                     [N/m**2]

        Properties Used:

    """  

    # Unpack inputs
    cl_max    = ca.aerodynamics.cl_max_landing
    Sg        = ca.analyses.landing.ground_roll
    altitude  = ca.analyses.landing.runway_elevation     
    eps       = ca.analyses.landing.approach_speed_factor
    delta_ISA = ca.analyses.landing.delta_ISA

    Nets = SUAVE.Components.Energy.Networks 

    # Estimate maximum lift coefficient
    if cl_max == 0:
        cl_max = compute_max_lift_constraint(vehicle.wings['main_wing'])

    # Estimate the approach speed
    for prop in vehicle.networks: 
        if isinstance(prop, Nets.Battery_Propeller) or isinstance(prop, Nets.Combustion_Propeller) or \
           isinstance(prop, Nets.Combustion_Propeller_Constant_Speed):
            kappa = 0.7
            Vapp  = np.sqrt(3.8217*Sg/kappa+49.488)

        elif isinstance(prop, Nets.Turbofan) or isinstance(prop, Nets.Turbojet_Super):
            kappa = 0.6
            Vapp  = np.sqrt(2.6319*Sg/kappa+458.8)

        else:
            raise ValueError('Warning: Specify an existing energy network')
        

    # Determine atmospheric properties at the altitude
    planet      = SUAVE.Analyses.Planets.Planet()
    atmosphere  = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features

    atmo_values = atmosphere.compute_values(altitude,delta_ISA)
    rho         = atmo_values.density[0,0]

    W_S        = np.zeros(2)
        
    cl_app     = cl_max / eps**2
    W_S[:]     = 0.5*rho*Vapp**2*cl_app


    return W_S

