## @ingroup Methods-Constraint_Analysis
# compute_turn_constraint.py
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
#  Compute turn constraint
# ----------------------------------------------------------------------

## @ingroup Methods-Constraint_Analysis
def compute_turn_constraint(ca,vehicle):
    
    """Calculate thrust-to-weight ratios for the turn maneuver

        Assumptions:
            Minimum drag coefficient is independent on the altitude and airspeed

        Source:
            S. Gudmundsson "General Aviation aircraft Design. Applied methods and procedures", Butterworth-Heinemann (6 Nov. 2013), ISBN - 0123973082

        Inputs:
            ca.analyses.turn.angle                [radians]
                        altitude                  [m]
                        delta_ISA                 [K]
                        airspeed                  [m/s]
                        specific_energy
                        thrust_fraction           [Unitless]
               wing_loading                       [N/m**2]

        Outputs:
            constraints.T_W                     [Unitless]
                        W_S                     [N/m**2]
                        
        Properties Used:
    """         

    # ==============================================
    # Unpack inputs
    # ==============================================
    phi       = ca.analyses.turn.angle         
    altitude  = ca.analyses.turn.altitude        
    delta_ISA = ca.analyses.turn.delta_ISA       
    M         = ca.analyses.turn.mach       
    Ps        = ca.analyses.turn.specific_energy 
    throttle  = ca.analyses.turn.thrust_fraction  
    W_S       = ca.wing_loading

    Nets  = SUAVE.Components.Energy.Networks 


    # Determine atmospheric properties at the altitude
    planet      = SUAVE.Analyses.Planets.Planet()
    atmosphere  = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features

    atmo_values = atmosphere.compute_values(altitude,delta_ISA)
    rho         = atmo_values.density[0,0]
    Vturn       = M * atmo_values.speed_of_sound[0,0]
    q           = 0.5*rho*Vturn**2  

    T_W = np.zeros(len(W_S))
    for prop in vehicle.networks: 
        if isinstance(prop, Nets.Battery_Propeller) or isinstance(prop, Nets.Combustion_Propeller) or \
           isinstance(prop, Nets.Combustion_Propeller_Constant_Speed):
        
            P_W  = np.zeros(len(W_S))
            etap = ca.propeller.turn_efficiency
            if etap == 0:
                raise ValueError('Warning: Set the propeller efficiency during turn')

        k,cd0 = compute_constraint_aero_values(W_S,M,q,vehicle,ca)
        T_W   = q * (cd0/W_S + k/(q*np.cos(phi))**2*W_S) + Ps/Vturn

        # Convert thrust to power (for propeller-driven engines) and normalize the results wrt the Sea-level
        if isinstance(prop, Nets.Battery_Propeller) or isinstance(prop, Nets.Combustion_Propeller) or \
           isinstance(prop, Nets.Combustion_Propeller_Constant_Speed):

            P_W = T_W*Vturn/etap

            if isinstance(prop, Nets.Combustion_Propeller) or isinstance(prop, Nets.Combustion_Propeller_Constant_Speed):
                if hasattr(prop.engines, "simple_turbomachine") is True:
                    P_W = P_W / normalize_turboprop_thrust(atmo_values) 
                elif hasattr(prop.engines, "internal_combustion_engine") is True:
                    P_W = P_W / normalize_power_piston(rho)
                else:
                    raise ValueError('Warning: Specify an existing engine type') 
            elif isinstance(prop, Nets.Battery_Propeller):
                pass 
            else:
                raise ValueError('Warning: Specify an existing engine type') 


        elif isinstance(prop, Nets.Turbofan) or isinstance(prop, Nets.Turbojet_Super):
            T_W = T_W / normalize_gasturbine_thrust(ca,vehicle,atmo_values,M*np.ones(1),'turn') 

        else:
            raise ValueError('Warning: Specify an existing energy network')


    # Pack outputs
    if isinstance(prop, Nets.Turbofan) or isinstance(prop, Nets.Turbojet_Super):
        constraint = T_W / throttle        
    else:
        constraint = P_W / throttle       # in W/N

    return constraint

