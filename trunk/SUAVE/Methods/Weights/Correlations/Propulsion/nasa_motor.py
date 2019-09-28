## @ingroup Methods-Weights-Correlations-Propulsion
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

## @ingroup Methods-Weights-Correlations-Propulsion
def nasa_motor(torque, kwt2=.3928, xwt=.8587):
    """ Calculate the weight of an air-cooled motor    
    weight correlation; weight=kwt2*(max_power**xwt)
        
    Assumptions:
            calculated from fit of high power-to-weight motors
            
    Source: NDARC Theory Manual
    
    Inputs:
            torque- maximum torque the motor can deliver safely      [N-m]
            kwt2
            xwt
            
    Outputs:
            weight- weight of the motor                                [kilograms]
        
    Properties Used:
            N/A
    """   
    trq  = torque/(Units.ft*Units.lbf)
    weight = kwt2*(trq**xwt) * Units.pounds #weight in lbs.
    
    return weight