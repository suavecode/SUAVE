## @ingroup Methods-Constraint_Analysis
# compute_climb_constraint.py
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
#  Compute climb constraint
# ----------------------------------------------------------------------

## @ingroup Methods-Constraint_Analysis
def compute_climb_constraint(ca,vehicle):
    
    """Calculate thrust-to-weight ratios for the steady climb

        Assumptions:
            N/A

        Source:
            S. Gudmundsson "General Aviation aircraft Design. Applied methods and procedures", Butterworth-Heinemann (6 Nov. 2013), ISBN - 0123973082

        Inputs:
                ca.analyses
                    climb.altitude                 [m]
                          airspeed                 [m/s]
                          climb_rate               [m/s]
                          delta_ISA                [K]
                   wing_loading             [N/m**2]
                 

        Outputs:
            constraints.T_W                     [Unitless]
                        W_S                     [N/m**2]

        Properties Used:
    """  

    # Unpack inputs 
    altitude  = ca.analyses.climb.altitude   
    Vx        = ca.analyses.climb.airspeed   
    Vy        = ca.analyses.climb.climb_rate                   
    delta_ISA = ca.analyses.climb.delta_ISA
    W_S       = ca.wing_loading 

    Nets  = SUAVE.Components.Energy.Networks 
  

    # Determine atmospheric properties at the altitude
    planet      = SUAVE.Analyses.Planets.Planet()
    atmosphere  = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features

    atmo_values = atmosphere.compute_values(altitude,delta_ISA)
    rho         = atmo_values.density[0,0]
    a           = atmo_values.speed_of_sound[0,0]   


    for prop in vehicle.networks: 
        if isinstance(prop, Nets.Battery_Propeller) or isinstance(prop, Nets.Internal_Combustion_Propeller) or \
           isinstance(prop, Nets.Internal_Combustion_Propeller_Constant_Speed) or isinstance(prop, Nets.Turboprop):
        
            T_W        = np.zeros(len(W_S))
            P_W        = np.zeros(len(W_S))
            etap       = ca.propeller.climb_efficiency
            if etap == 0:
                raise ValueError('Warning: Set the propeller efficiency during climb')

            for i in range(len(W_S)):
                error      = 1
                tolerance = 1e-6
                M          = 0.5                            # Initial Mach number
                q          = 0.5*rho*(M*a)**2               # Initial dynamic pressure
                # Iterate the best propeller climb speed estimation until converged 
                while abs(error) > tolerance:
 
                    k,cd0 = compute_constraint_aero_values(W_S[i],M,q,vehicle,ca)
                    Vx    = np.sqrt(2/rho*W_S[i]*np.sqrt(k/(3*cd0)))
                    Mnew  = Vx/a
                    error = Mnew - M
                    M     = Mnew
                    q     = 0.5*rho*(M*a)**2 

                T_W[i] = Vy/Vx + q/W_S[i]*cd0 + k/q*W_S[i]
                P_W[i] = T_W[i]*Vx/etap


                if isinstance(prop, Nets.Turboprop):
                    P_W[i] = P_W[i] / normalize_turboprop_thrust(atmo_values) 
                elif isinstance(prop, Nets.Internal_Combustion_Propeller) or isinstance(prop, Nets.Internal_Combustion_Propeller_Constant_Speed):
                    P_W[i] = P_W[i] / normalize_power_piston(rho)
                elif isinstance(prop, Nets.Battery_Propeller):
                    pass 

        elif isinstance(prop, Nets.Turbofan) or isinstance(prop, Nets.Turbojet_Super):
            M  = Vx/a * np.ones(1)                          
            q  = 0.5*rho*Vx**2      

            k,cd0  = compute_constraint_aero_values(W_S,M,q,vehicle,ca)
            T_W = (Vy/Vx + q/W_S*cd0 + k/q*W_S) / normalize_gasturbine_thrust(ca,vehicle,atmo_values,M,'climb')

        else:
            raise ValueError('Warning: Specify an existing energy network')

    # Pack outputs
    if isinstance(prop, Nets.Turbofan) or isinstance(prop, Nets.Turbojet_Super):
        constraint = T_W         
    else:
        constraint = P_W        # in W/N

    return constraint

