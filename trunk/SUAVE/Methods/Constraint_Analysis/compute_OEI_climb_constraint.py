## @ingroup Methods-Constraint_Analysis
# compute_OEI_climb_constraint.py
#
# Created:  Nov 2021, S. Karpuk, 
# Modified: 



# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE Imports
import SUAVE
from SUAVE.Core import Units
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Helper_Functions.oswald_efficiency  import oswald_efficiency               as  oswald_efficiency
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
def compute_OEI_climb_constraint(vehicle):
    
    """Calculate thrust-to-weight ratios for the 2nd segment OEI climb

        Assumptions:
            Climb CL and CD are similar to the take-off CL and CD
            Based on FAR Part 25

        Source:
            L. Loftin,Subsonic Aircraft: Evolution and the Matching of Size to Performance, NASA Ref-erence Publication 1060, August 1980

        Inputs:
            self.engine.type                    string
                 takeoff.runway_elevation       [m]
                         delta_ISA              [K]
                 aerodynamics.cd_takeoff        [Unitless]
                              cl_takeoff        [Unitless]
                OEI_climb.climb_speed_factor    [Unitless]
                engine.number                   [Unitless]
                wing_loading                    [N/m**2]

        Outputs:
            constraints.T_W                     [Unitless]
                        W_S                     [N/m**2]

        Properties Used:

    """   

    # Unpack inputs
    eng_type  = vehicle.constraints.engine.type
    altitude  = vehicle.constraints.analyses.takeoff.runway_elevation     
    delta_ISA = vehicle.constraints.analyses.takeoff.delta_ISA   
    cd_TO     = vehicle.constraints.aerodynamics.cd_takeoff 
    cl_TO     = vehicle.constraints.aerodynamics.cl_takeoff
    eps       = vehicle.constraints.analyses.OEI_climb.climb_speed_factor 
    Ne        = vehicle.constraints.engine.number
    W_S       = vehicle.constraints.wing_loading 

    # determine the flight path angle
    if Ne == 2:
        gamma = 1.203 * Units.degrees
    elif Ne == 3:
        gamma = 1.3748 * Units.degrees
    elif Ne == 4:
        gamma = 1.5466 * Units.degrees
    else:
        gamma = 0.0

    # Determine atmospheric properties at the altitude
    planet      = SUAVE.Analyses.Planets.Planet()
    atmosphere  = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features

    atmo_values = atmosphere.compute_values(altitude,delta_ISA)
    rho         = atmo_values.density[0,0]
    a           = atmo_values.speed_of_sound[0,0] 
    Vcl         = eps * np.sqrt(2*W_S/(rho*cl_TO))
    M           = Vcl/a

    L_D = cl_TO/cd_TO

    T_W    = np.zeros(len(W_S))
    T_W[:] = Ne/((Ne-1)*L_D)+gamma

    if eng_type != ('turbofan' or 'Turbofan') and eng_type != ('turbojet' or 'Turbojet'):
        P_W  = np.zeros(len(W_S))
        etap = vehicle.constraints.propeller.cruise_efficiency
        if etap == 0:
            raise ValueError('Warning: Set the propeller efficiency during the OEI 2nd climb segment')

        for i in range(len(W_S)):
            P_W[i] = T_W[i]*Vcl[i]/etap

            if eng_type   == ('turboprop' or 'Turboprop'):
                P_W[i] = P_W[i] / normalize_turboprop_thrust(atmo_values)
            elif eng_type == ('piston' or 'Piston'):
                P_W[i] = P_W[i] / normalize_power_piston(rho) 
            elif eng_type == ('electric air-cooled' or 'Electric air-cooled'):
                P_W[i] = P_W[i] / normalize_power_electric(rho)  
            elif eng_type == ('electric liquid-cooled' or 'Electric liquid-cooled'):
                pass 
    else:
        for i in range(len(W_S)):
            T_W[i] = T_W[i] / normalize_gasturbine_thrust(vehicle,atmo_values,M[i],'OEIclimb')  

    # Pack outputs
    if eng_type != ('turbofan' or 'Turbofan') and eng_type != ('turbojet' or 'Turbojet'):
        constraint = P_W         # convert to W/N
    else:
        constraint = T_W

    return constraint

