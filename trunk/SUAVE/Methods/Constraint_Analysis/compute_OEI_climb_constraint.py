## @ingroup Methods-Constraint_Analysis
# compute_OEI_climb_constraint.py
#
# Created:  Nov 2021, S. Karpuk, 
# Modified: 



# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE Imports
import calendar
from struct import calcsize
import SUAVE
from SUAVE.Core import Units
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Helper_Functions.oswald_efficiency  import oswald_efficiency               as  oswald_efficiency
from SUAVE.Methods.Constraint_Analysis.normalize_propulsion                              import normalize_turboprop_thrust
from SUAVE.Methods.Constraint_Analysis.normalize_propulsion                              import normalize_power_piston
from SUAVE.Methods.Constraint_Analysis.normalize_propulsion                              import normalize_gasturbine_thrust


# package imports
import numpy as np

# ----------------------------------------------------------------------
#  Compute turn constraint
# ----------------------------------------------------------------------

## @ingroup Methods-Constraint_Analysis
def compute_OEI_climb_constraint(ca,vehicle):
    
    """Calculate thrust-to-weight ratios for the 2nd segment OEI climb

        Assumptions:
            Climb CL and CD are similar to the take-off CL and CD
            Based on FAR Part 25

        Source:
            L. Loftin,Subsonic Aircraft: Evolution and the Matching of Size to Performance, NASA Ref-erence Publication 1060, August 1980

        Inputs:
            ca.analyses.takeoff.runway_elevation         [m]
                                delta_ISA                [K]
                        OEI_climb.climb_speed_factor    [Unitless]

               aerodynamics.cd_takeoff         [Unitless]
                            cl_takeoff         [Unitless]
               wing_loading                    [N/m**2]

        Outputs:
            constraints.T_W                     [Unitless]
                        W_S                     [N/m**2]

        Properties Used:

    """   

    # Unpack inputs
    altitude  = ca.analyses.takeoff.runway_elevation     
    delta_ISA = ca.analyses.takeoff.delta_ISA   
    cd_TO     = ca.aerodynamics.cd_takeoff 
    cl_TO     = ca.aerodynamics.cl_takeoff
    eps       = ca.analyses.OEI_climb.climb_speed_factor 
    W_S       = ca.wing_loading 

    for prop in vehicle.networks:
        Ne = prop.number_of_engines

    Nets  = SUAVE.Components.Energy.Networks 

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

    for prop in vehicle.networks: 
        if isinstance(prop, Nets.Battery_Propeller) or isinstance(prop, Nets.Internal_Combustion_Propeller) or \
           isinstance(prop, Nets.Internal_Combustion_Propeller_Constant_Speed) or isinstance(prop, Nets.Turboprop):
        
            P_W  = np.zeros(len(W_S))
            etap = ca.propeller.cruise_efficiency
            if etap == 0:
                raise ValueError('Warning: Set the propeller efficiency during the OEI 2nd climb segment')

            P_W = T_W*Vcl/etap

            if isinstance(prop, Nets.Turboprop):
                P_W = P_W / normalize_turboprop_thrust(atmo_values) 
            elif isinstance(prop, Nets.Internal_Combustion_Propeller) or isinstance(prop, Nets.Internal_Combustion_Propeller_Constant_Speed):
                P_W = P_W / normalize_power_piston(rho)
            elif isinstance(prop, Nets.Battery_Propeller):
                pass 

        elif isinstance(prop, Nets.Turbofan) or isinstance(prop, Nets.Turbojet_Super):
            T_W = T_W / normalize_gasturbine_thrust(ca,vehicle,atmo_values,M*np.zeros(1),'OEIclimb')  

        else:
            raise ValueError('Warning: Specify an existing energy network')    

    # Pack outputs
    if isinstance(prop, Nets.Turbofan) or isinstance(prop, Nets.Turbojet_Super):
        constraint = T_W        
    else:
        constraint = P_W         # convert to W/N

    return constraint

