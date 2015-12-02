# air_cooled_motor.py
# 
# Created:  Michael Vegh, Jan 2014
# Modified: Andrew Wendorff, Jan 2014        


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units
from SUAVE.Core import (
    Data, Container, Data_Exception, Data_Warning,
)

# ----------------------------------------------------------------------
#   Method
# ----------------------------------------------------------------------


def air_cooled_motor(max_power, kwt2=1.96, xwt=.8897):
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
                
                Source: Sinsay, J.D., Tracey, B., Alonso, J.J., Kontinos, D.K., Melton, J.E., Grabbe, S.,
                "Air Vehicle Design and Technology Considerations for an Electric VTOL Metro-Regional Public Transportation System,"
                12th AIAA Aviation Technology, Integration, and Operations (ATIO) Conference and 14th AIAA/ISSMO Multidisciplinary
                Analysis and Optimization Conference, Indianapolis, IN, Sept.2012
    """    
    
    
    # process
    weight = kwt2*((max_power/Units.kW)**xwt)   #weight in lbs.
    mass   =weight/Units.kg
    return mass