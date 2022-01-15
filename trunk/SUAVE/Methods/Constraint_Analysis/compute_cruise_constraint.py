## @ingroup Methods-Constraint_Analysis
# compute_cruise_constraint.py
#
# Created:  Nov 2021, S. Karpuk, 
# Modified: 



# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE Imports
import SUAVE
from SUAVE.Methods.Constraint_Analysis.compute_constraint_aero_values                    import compute_constraint_aero_values  
from SUAVE.Methods.Constraint_Analysis.normalize_propulsion                              import normalize_turboprop_thrust
from SUAVE.Methods.Constraint_Analysis.normalize_propulsion                              import normalize_power_piston
from SUAVE.Methods.Constraint_Analysis.normalize_propulsion                              import normalize_gasturbine_thrust


# package imports
import numpy as np

# ----------------------------------------------------------------------
#  Compute cruise constraint
# ----------------------------------------------------------------------

## @ingroup Methods-Constraint_Analysis
def compute_cruise_constraint(ca,vehicle,cruise_tag):
    
    """Calculate thrust-to-weight ratios for the cruise

        Assumptions:
           N/A 

        Source:
            S. Gudmundsson "General Aviation aircraft Design. Applied methods and procedures", Butterworth-Heinemann (6 Nov. 2013), ISBN - 0123973082

        Inputs:
            ca.analyses.cruise.altitude                     [m]
                               delta_ISA                    [K]
                               airspeed                     [m/s]
                               thrust_fraction              [Unitless]
               wing_loading                        [N/m**2]


        Outputs:
            constraints.T_W                     [Unitless]
                        W_S                     [N/m**2]

        Properties Used:
    """  

    # Unpack inputs
    if cruise_tag == 'max cruise':
        altitude    = ca.analyses.max_cruise.altitude 
        delta_ISA   = ca.analyses.max_cruise.delta_ISA
        M           = ca.analyses.max_cruise.mach 
        throttle    = ca.analyses.max_cruise.thrust_fraction 
    else:
        altitude    = ca.analyses.cruise.altitude 
        delta_ISA   = ca.analyses.cruise.delta_ISA
        M           = ca.analyses.cruise.mach 
        throttle    = ca.analyses.cruise.thrust_fraction

    W_S   = ca.wing_loading

    Nets  = SUAVE.Components.Energy.Networks      

    # Determine atmospheric properties at the altitude
    planet      = SUAVE.Analyses.Planets.Planet()
    atmosphere  = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features

    atmo_values = atmosphere.compute_values(altitude,delta_ISA)
    rho         = atmo_values.density[0,0]
    Vcr         = M * atmo_values.speed_of_sound[0,0]
    q           = 0.5*rho*Vcr**2

    T_W = np.zeros(len(W_S))
    for prop in vehicle.networks: 
        if isinstance(prop, Nets.Battery_Propeller) or isinstance(prop, Nets.Internal_Combustion_Propeller) or \
           isinstance(prop, Nets.Internal_Combustion_Propeller_Constant_Speed) or isinstance(prop, Nets.Turboprop):
            P_W  = np.zeros(len(W_S))
            etap = ca.propeller.cruise_efficiency
            if etap == 0:
                raise ValueError('Warning: Set the propeller efficiency during cruise')

        k,cd0  = compute_constraint_aero_values(W_S,M,q,vehicle,ca)
        T_W    = q*cd0/W_S+k*W_S/q

        # Convert thrust to power (for propeller-driven engines) and normalize the results wrt the Sea-level
        if isinstance(prop, Nets.Battery_Propeller) or isinstance(prop, Nets.Internal_Combustion_Propeller) or \
           isinstance(prop, Nets.Internal_Combustion_Propeller_Constant_Speed) or isinstance(prop, Nets.Turboprop):

            P_W = T_W*Vcr/etap

            if isinstance(prop, Nets.Turboprop):
                P_W = P_W / normalize_turboprop_thrust(atmo_values) 
            elif isinstance(prop, Nets.Internal_Combustion_Propeller) or isinstance(prop, Nets.Internal_Combustion_Propeller_Constant_Speed):
                P_W = P_W / normalize_power_piston(rho)
            elif isinstance(prop, Nets.Battery_Propeller):
                pass 

        elif isinstance(prop, Nets.Turbofan) or isinstance(prop, Nets.Turbojet_Super):
            T_W = T_W / normalize_gasturbine_thrust(ca,vehicle,atmo_values,M*np.ones(1),'cruise')   

        else:
            raise ValueError('Warning: Specify an existing energy network')      

    # Pack outputs
    if isinstance(prop, Nets.Turbofan) or isinstance(prop, Nets.Turbojet_Super):
        constraint = T_W / throttle        
    else:
        constraint = P_W / throttle        # in W/N

        
    return constraint

