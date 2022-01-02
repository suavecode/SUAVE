## @ingroup Methods-Constraint_Analysis
# compute_take_off_constraint.py
#
# Created:  Nov 2021, S. Karpuk, 
# Modified: 



# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE Imports
import SUAVE
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Helper_Functions.oswald_efficiency  import oswald_efficiency            as  oswald_efficiency
from SUAVE.Methods.Constraint_Analysis.normalize_propulsion                              import normalize_turboprop_thrust
from SUAVE.Methods.Constraint_Analysis.normalize_propulsion                              import normalize_power_piston
from SUAVE.Methods.Constraint_Analysis.normalize_propulsion                              import normalize_power_electric
from SUAVE.Methods.Constraint_Analysis.normalize_propulsion                              import normalize_gasturbine_thrust
from SUAVE.Methods.Constraint_Analysis.normalize_propulsion                              import normalize_gasturbine_thrust
from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift.compute_max_lift_constraint           import compute_max_lift_constraint


# package imports
import numpy as np

# ----------------------------------------------------------------------
#  Compute take-off constraint
# ----------------------------------------------------------------------

## @ingroup Methods-Constraint_Analysis
def compute_take_off_constraint(vehicle):
    
    """Calculate thrust-to-weight ratios at take-off

        Assumptions:
            Maximum take-off lift coefficient is 85% of the maximum landing lift coefficient
            
        Source:
            S. Gudmundsson "General Aviation aircraft Design. Applied methods and procedures", Butterworth-Heinemann (6 Nov. 2013), ISBN - 0123973082

        Inputs:
            constraint.engine.type                        string
                 aerodynamics.cd_takeoff            [Unitless]
                              cl_takeoff            [Unitless]
                              cl_max_takeoff        [Unitless]
                takeoff.ground_run                  [m]      
                        liftoff_speed_factor        [Unitless]
                        rolling_resistance
                        runway_elevation            [m]
                        delta_ISA                   [K]
                        
                wing_loading                        [N/m**2]
                
        Outputs:
            constraints.T_W                         [Unitless]
                        W_S                         [W/N]

        Properties Used:

    """       

    # ==============================================
    # Unpack inputs
    # ==============================================
    eng_type  = vehicle.constraints.engine.type
    cd_TO     = vehicle.constraints.aerodynamics.cd_takeoff 
    cl_TO     = vehicle.constraints.aerodynamics.cl_takeoff 
    cl_max_TO = vehicle.constraints.aerodynamics.cl_max_takeoff
    Sg        = vehicle.constraints.analyses.takeoff.ground_run
    eps       = vehicle.constraints.analyses.takeoff.liftoff_speed_factor
    miu       = vehicle.constraints.analyses.takeoff.rolling_resistance
    altitude  = vehicle.constraints.analyses.takeoff.runway_elevation
    delta_ISA = vehicle.constraints.analyses.takeoff.delta_ISA
    W_S       = vehicle.constraints.wing_loading


    # Set take-off aerodynamic properties
    if cl_TO == 0 or cd_TO == 0:
        raise ValueError("Define cl_takeoff or cd_takeoff\n")

    if cl_max_TO == 0:
        cl_max_LD = compute_max_lift_constraint(vehicle.wings['main_wing'])     # Landing maximum lift coefficient     
        cl_max_TO = 0.85 * cl_max_LD                                                                    # Take-off flaps settings

    # Check if the take-off distance was input
    if Sg == 0:
        raise ValueError("Input the ground_run distance\n")
        
    # Determine atmospheric properties at the altitude
    planet      = SUAVE.Analyses.Planets.Planet()
    atmosphere  = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features

    atmo_values = atmosphere.compute_values(altitude,delta_ISA)
    rho         = atmo_values.density[0,0]
    g           = atmosphere.planet.sea_level_gravity


    T_W = np.zeros(len(W_S))
    if eng_type != ('turbofan' or 'Turbofan') and eng_type != ('turbojet' or 'Turbojet'):
        P_W  = np.zeros(len(W_S))
        etap = vehicle.constraints.propeller.takeoff_efficiency
        if etap == 0:
            raise ValueError('Warning: Set the propeller efficiency during take-off')
    for i in range(len(W_S)):
        Vlof   = eps*np.sqrt(2*W_S[i]/(rho*cl_max_TO))
        Mlof   = Vlof/atmo_values.speed_of_sound[0,0]
        T_W[i] = eps**2*W_S[i] / (g*Sg*rho*cl_max_TO) + eps*cd_TO/(2*cl_max_TO) + miu * (1-eps*cl_TO/(2*cl_max_TO))
            
        # Convert thrust to power (for propeller-driven engines) and normalize the results wrt the Sea-level
        if eng_type != ('turbofan' or 'Turbofan') and eng_type != ('turbojet' or 'Turbojet'):
                
            P_W[i] = T_W[i]*Vlof/etap

            if eng_type == ('turboprop' or 'Turboprop'):
                P_W[i] = P_W[i] / normalize_turboprop_thrust(atmo_values) 
            elif eng_type == ('piston' or 'Piston'):
                P_W[i] = P_W[i] / normalize_power_piston(rho)
            elif eng_type == ('electric air-cooled' or 'Electric air-cooled'):
                P_W[i] = P_W[i] / normalize_power_electric(rho)  
            elif eng_type == ('electric liquid-cooled' or 'Electric liquid-cooled'):
                pass 
        else:
            T_W[i] = T_W[i] / normalize_gasturbine_thrust(vehicle,atmo_values,Mlof,'takeoff')  
     

    # Pack outputs
    if eng_type != ('turbofan' or 'Turbofan') and eng_type != ('turbojet' or 'Turbojet'):
        constraint = P_W         # convert to W/N
    else:
        constraint = T_W

    return constraint

