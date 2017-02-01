# air_cooled_motor.py
# 
# Created:  Jan 2014, M. Vegh, 
# Modified: Jan 2014, A. Wendorff
#           Feb 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units

# ----------------------------------------------------------------------
#   Air Cooled Motor
# ----------------------------------------------------------------------

def engine_piston(max_power, kwt2=5.22, xwt=.780):
    """ weight = SUAVE.Methods.Correlations.Propulsion.air_cooled_motor(max_power)
        Calculate the weight of an air-cooled motor    
        weight correlation; weight=kwt2*(max_power**xwt)
        Inputs:
                max_power- maximum power the motor can deliver safely [Watts]
                kwt2
                xwt
                
        Outputs:
                weight- weight of the motor [kilograms]
            
        Assumptions:
                calculated from fit of commercial available motors
                
                Source: Raymer, Aircraft Design, a Conceptual Approach
    """    
    bhp    = max_power/Units.horsepower
    weight = kwt2*((bhp)**xwt)  #weight in lbs.
    mass   = weight*Units.lbs
    return mass