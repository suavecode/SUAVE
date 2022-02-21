## @ingroup Methods-Performance
# estimate_stall_speed.py
#
# Created: Jan 2022, M. Clarke

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------
import SUAVE
import numpy as np 

#------------------------------------------------------------------------------
# Stall Speed Function
#------------------------------------------------------------------------------

## @ingroup Methods-Performance
def estimate_stall_speed(vehicle_mass,reference_area,altitude,maximum_lift_coefficient):

    """Calculates the stall speed of an aircraft at a given altitude and a maximum lift coefficient.

        Sources:
        N/A

        Assumptions:
        None 

        Inputs:
            vehicle_mass                    vehicle mass             [kg]
            reference_area                  vehicle reference area   [m^2] 
            altitude                        cruise altitude          [m]
            maximum_lift_coefficient        maximum lift coefficient [unitless] 
            
        Outputs: 
            V_stall                         stall speed              [m/s]
    """ 
      
    g       = 9.81 
    atmo    = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    rho     = atmo.compute_values(altitude,0.).density 
    V_stall = float(np.sqrt(2.*vehicle_mass*g/(rho*reference_area*maximum_lift_coefficient)))  
    
    return V_stall