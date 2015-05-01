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


def air_cooled_motor(max_power):
    """ weight = SUAVE.Methods.Correlations.Propulsion.air_cooled_motor(max_power)
        Calculate the weight of an air-cooled motor    
    
        Inputs:
                max_power- maximum power the motor can deliver safely [Watts]
        
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
    weight = ((1./2.2)*1.96*((max_power/1000.)**.8897))   #weight in kg
    
    return weight