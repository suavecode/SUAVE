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
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Helper_Functions.oswald_efficiency  import oswald_efficiency                   as  oswald_efficiency
from SUAVE.Methods.Constraint_Analysis.compressibility_drag_constraint                   import compressibility_drag_constraint     as  compressibility_drag
from SUAVE.Methods.Constraint_Analysis.normalize_propulsion                              import normalize_turboprop_thrust
from SUAVE.Methods.Constraint_Analysis.normalize_propulsion                              import normalize_power_piston
from SUAVE.Methods.Constraint_Analysis.normalize_propulsion                              import normalize_power_electric
from SUAVE.Methods.Constraint_Analysis.normalize_propulsion                              import normalize_gasturbine_thrust


# package imports
import numpy as np

# ----------------------------------------------------------------------
#  Compute turn constraint
# ----------------------------------------------------------------------

## @ingroup Methods-Constraint_Analysis
def compute_turn_constraint(vehicle):
    
    """Calculate thrust-to-weight ratios for the turn maneuver

        Assumptions:
            Minimum drag coefficient is independent on the altitude and airspeed

        Source:
            S. Gudmundsson "General Aviation aircraft Design. Applied methods and procedures", Butterworth-Heinemann (6 Nov. 2013), ISBN - 0123973082

        Inputs:
            self.aerodynamics.cd_min_clean      [Unitless]
                 engine.type                    string
                 turn.angle                     [radians]
                      altitude                  [m]
                      delta_ISA                 [K]
                      airspeed                  [m/s]
                      specific_energy
                      thrust_fraction           [Unitless]
                 geometry.aspect_ratio          [Unitless]
                 wing_loading                   [N/m**2]

        Outputs:
            constraints.T_W                     [Unitless]
                        W_S                     [N/m**2]
                        
        Properties Used:
    """         

    # ==============================================
    # Unpack inputs
    # ==============================================
    cd_min    = vehicle.constraints.aerodynamics.cd_min_clean 
    eng_type  = vehicle.constraints.engine.type
    phi       = vehicle.constraints.analyses.turn.angle         
    altitude  = vehicle.constraints.analyses.turn.altitude        
    delta_ISA = vehicle.constraints.analyses.turn.delta_ISA       
    M         = vehicle.constraints.analyses.turn.mach       
    Ps        = vehicle.constraints.analyses.turn.specific_energy 
    throttle  = vehicle.constraints.analyses.turn.thrust_fraction   
    AR        = vehicle.wings['main_wing'].aspect_ratio
    W_S       = vehicle.constraints.wing_loading


    # Determine atmospheric properties at the altitude
    planet      = SUAVE.Analyses.Planets.Planet()
    atmosphere  = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features

    atmo_values = atmosphere.compute_values(altitude,delta_ISA)
    rho         = atmo_values.density[0,0]
    Vturn       = M * atmo_values.speed_of_sound[0,0]
    q           = 0.5*rho*Vturn**2  

    T_W = np.zeros(len(W_S))
    if eng_type != ('turbofan' or 'Turbofan') and eng_type != ('turbojet' or 'Turbojet'):
        P_W  = np.zeros(len(W_S))
        etap = vehicle.constraints.propeller.turn_efficiency
        if etap == 0:
            raise ValueError('Warning: Set the propeller efficiency during turn')

    for i in range(len(W_S)):
        CL = W_S[i]/q

        # Calculate compressibility_drag
        cd_comp   = compressibility_drag(M,CL,vehicle.wings['main_wing']) 
        cd_comp_e = compressibility_drag(M,0,vehicle.wings['main_wing'])  
        cd0       = cd_min + cd_comp  

        # Calculate Oswald efficiency
        e = oswald_efficiency(vehicle,cd_min+cd_comp_e)

        k      = 1/(np.pi*e*AR)
        T_W[i] = q * (cd0/W_S[i] + k/(q*np.cos(phi))**2*W_S[i]) + Ps/Vturn

    # Convert thrust to power (for propeller-driven engines) and normalize the results wrt the Sea-level
    if eng_type != ('turbofan' or 'Turbofan') and eng_type != ('turbojet' or 'Turbojet'):

        P_W = T_W*Vturn/etap

        if eng_type == ('turboprop' or 'Turboprop'):
            P_W = P_W / normalize_turboprop_thrust(atmo_values) 
        elif eng_type == 'piston':
            P_W = P_W / normalize_power_piston(rho) 
        elif eng_type == ('electric air-cooled' or 'Electric air-cooled'):
            P_W = P_W / normalize_power_electric(rho)  
        elif eng_type == ('electric liquid-cooled' or 'Electric liquid-cooled'):
            pass 
    else:
        T_W = T_W / normalize_gasturbine_thrust(vehicle,atmo_values,M,'turn')          

    # Pack outputs
    if eng_type != ('turbofan' or 'Turbofan') and eng_type != ('turbojet' or 'Turbojet'):
        constraint = P_W / throttle         # convert to W/N
    else:
        constraint = T_W / throttle 

    return constraint

