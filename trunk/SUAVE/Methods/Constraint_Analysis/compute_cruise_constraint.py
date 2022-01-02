## @ingroup Methods-Constraint_Analysis
# compute_cruise_constraint.py
#
# Created:  Nov 2021, S. Karpuk, 
# Modified: 



# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUave Imports
import SUAVE
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Helper_Functions.oswald_efficiency  import oswald_efficiency                   as  oswald_efficiency
from SUAVE.Methods.Constraint_Analysis.compressibility_drag_constraint                   import compressibility_drag_constraint     as  compressibility_drag
from SUAVE.Methods.Constraint_Analysis.normalize_propulsion                              import normalize_turboprop_thrust
from SUAVE.Methods.Constraint_Analysis.normalize_propulsion                              import normalize_power_piston
from SUAVE.Methods.Constraint_Analysis.normalize_propulsion                              import normalize_power_electric
from SUAVE.Methods.Constraint_Analysis.normalize_propulsion                              import normalize_gasturbine_thrust


# package imports
import numpy as np

# ----------------------------------------------------------------------
#  Compute cruise constraint
# ----------------------------------------------------------------------

## @ingroup Methods-Constraint_Analysis
def compute_cruise_constraint(vehicle,cruise_tag):
    
    """Calculate thrust-to-weight ratios for the cruise

        Assumptions:
           N/A 

        Source:
            S. Gudmundsson "General Aviation aircraft Design. Applied methods and procedures", Butterworth-Heinemann (6 Nov. 2013), ISBN - 0123973082

        Inputs:
            self.engine.type                    string
            aerodynamics.cd_min_clean           [Unitless]
            cruise.altitude                     [m]
                   delta_ISA                    [K]
                   airspeed                     [m/s]
                   thrust_fraction              [Unitless]
            geometry.aspect_ratio               [Unitless]
            wing_loading                        [N/m**2]

        Outputs:
            constraints.T_W                     [Unitless]
                        W_S                     [N/m**2]

        Properties Used:
    """  

    # Unpack inputs
    if cruise_tag == 'max cruise':
        altitude    = vehicle.constraints.analyses.max_cruise.altitude 
        delta_ISA   = vehicle.constraints.analyses.max_cruise.delta_ISA
        M           = vehicle.constraints.analyses.max_cruise.mach 
        throttle    = vehicle.constraints.analyses.max_cruise.thrust_fraction 
    else:
        altitude    = vehicle.constraints.analyses.cruise.altitude 
        delta_ISA   = vehicle.constraints.analyses.cruise.delta_ISA
        M           = vehicle.constraints.analyses.cruise.mach 
        throttle    = vehicle.constraints.analyses.cruise.thrust_fraction

    eng_type  = vehicle.constraints.engine.type
    cd_min    = vehicle.constraints.aerodynamics.cd_min_clean 
    W_S       = vehicle.constraints.wing_loading
    AR        = vehicle.wings['main_wing'].aspect_ratio
              
    # Determine atmospheric properties at the altitude
    planet      = SUAVE.Analyses.Planets.Planet()
    atmosphere  = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features

    atmo_values = atmosphere.compute_values(altitude,delta_ISA)
    rho         = atmo_values.density[0,0]
    Vcr         = M * atmo_values.speed_of_sound[0,0]
    q           = 0.5*rho*Vcr**2

    T_W = np.zeros(len(W_S))
    if eng_type != 'turbofan' and eng_type != 'turbojet':
        P_W  = np.zeros(len(W_S))
        etap = vehicle.constraints.propeller.cruise_efficiency
        if etap == 0:
            raise ValueError('Warning: Set the propeller efficiency during cruise')
    for i in range(len(W_S)):
        CL = W_S[i]/q

        # Calculate compressibility_drag
        cd_comp   = compressibility_drag(M,CL,vehicle.wings['main_wing']) 
        cd_comp_e = compressibility_drag(M,0,vehicle.wings['main_wing'])  
        cd0       = cd_min + cd_comp  

        # Calculate Oswald efficiency
        e = oswald_efficiency(vehicle,cd_min+cd_comp_e)

        k = 1/(np.pi*e*AR)
        T_W[i] = q*cd0/W_S[i]+k*W_S[i]/q

    # Convert thrust to power (for propeller-driven engines) and normalize the results wrt the Sea-level
    if eng_type != ('turbofan' or 'Turbofan') and eng_type != ('turbojet' or 'Turbojet'):

        P_W = T_W*Vcr/etap

        if eng_type == ('turboprop' or 'Turboprop'):
            P_W[i] = P_W[i] / normalize_turboprop_thrust(atmo_values) 
        elif eng_type == ('piston' or 'Piston'):
            P_W[i] = P_W[i] / normalize_power_piston(rho)   
        elif eng_type == ('electric air-cooled' or 'Electric air-cooled'):
            P_W[i] = P_W[i] / normalize_power_electric(rho)  
        elif eng_type == ('electric liquid-cooled' or 'Electric liquid-cooled'):
            pass 
    else:
        T_W = T_W / normalize_gasturbine_thrust(vehicle,atmo_values,M,'cruise')     

    # Pack outputs
    if eng_type != ('turbofan' or 'Turbofan') and eng_type != ('turbojet' or 'Turbojet'):
        constraint = P_W / throttle         # convert to W/N
    else:
        constraint = T_W / throttle 

        
    return constraint

