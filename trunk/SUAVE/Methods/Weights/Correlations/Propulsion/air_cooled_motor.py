## @ingroup Methods-Weights-Correlations-Propulsion
# air_cooled_motor.py
# 
# Created:  Jan 2014, M. Vegh, 
# Modified: Jan 2014, A. Wendorff
#           Feb 2016, E. Botero
#           Mar 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units

# ----------------------------------------------------------------------
#   Air Cooled Motor
# ----------------------------------------------------------------------

## @ingroup Methods-Weights-Correlations-Propulsion
def air_cooled_motor(max_power, kwt2=1.96, xwt=.8897):
    """ Calculate the weight of an air-cooled motor    
    weight correlation; weight=kwt2*(max_power**xwt)
        
    Assumptions:
            calculated from fit of commercial available motors
            
    Source: Sinsay, J.D., Tracey, B., Alonso, J.J., Kontinos, D.K., Melton, J.E., Grabbe, S.,
            "Air Vehicle Design and Technology Considerations for an Electric VTOL Metro-Regional Public Transportation System,"
            12th AIAA Aviation Technology, Integration, and Operations (ATIO) Conference and 14th AIAA/ISSMO Multidisciplinary
            Analysis and Optimization Conference, Indianapolis, IN, Sept.2012  
    
    Inputs:
            max_power- maximum power the motor can deliver safely      [Watts]
            kwt2
            xwt
            
    Outputs:
            mass- weight of the motor                                [kilograms]
        
    Properties Used:
            N/A
    """   
    
    mass = kwt2*((max_power/Units.kW)**xwt) * Units.pounds # mass in kg 
    
    return mass