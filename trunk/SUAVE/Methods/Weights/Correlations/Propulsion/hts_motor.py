# hts_motor.py
# 
# Created:  Michael Vegh, Jan 2014
# Modified: Andrew Wendorff, Feb 2014         


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

def hts_motor(max_power):
    """ weight = SUAVE.Methods.Correlations.Propulsion.hts_motor(max_power)
        Calculate the weight of a high temperature superconducting motor
    
        Inputs:
                max_power- maximum power the motor can deliver safely [Watts]
        
        Outputs:
                weight- weight of the motor [kilograms]
            
        Assumptions:
                calculated from fit of commercial available motors
                
                Source: [10] Snyder, C., Berton, J., Brown, G. et all
                'Propulsion Investigation for Zero and Near-Zero Emissions Aircraft,' NASA STI Program,
                NASA Glenn,  2009.012
    """   

    # process
    weight=(1./2.2)*2.28*((max_power/1000.)**.6616)  #weight in kg
    
    return weight