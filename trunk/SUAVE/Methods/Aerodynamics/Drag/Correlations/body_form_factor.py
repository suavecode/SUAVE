# body_form_factor.py
#
# Created: April 2014, Tim Momose

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
import numpy as np
from SUAVE.Attributes import Units as Units
from SUAVE.Structure import (
    Data, Container, Data_Exception, Data_Warning,
)

# ----------------------------------------------------------------------
#  Method
# ----------------------------------------------------------------------

def body_form_factor(fineness_ratio,Mach):
    """ ff = SUAVE.Methods.Aerodynamics.Drag.Correlations.body_form_factor(fineness_ratio,Mach)
        This method computes the parasitic drag form factor of a non-lifting body
        
        Inputs:
            fineness_ratio - the fineness ratio of the equivalent body of
            revolution (If non-circular, take sqrt(4*S/pi) as the effective
            diameter, where S is the max cross-sectional area)[dimensionless]
            
            Mach - the flight Mach number at which form factor is evaluated
              
        Outputs:
            ff - the form factor of the body
                           
        Assumptions:
            - Uses Professor Ilan Kroo's correlation from the AA241 Course Notes.
    """             
    # Unpack inputs
    d = 1.0/fineness_ratio
    M = Mach
    
    #Compute form factor
    if M > 1.0:
        du = 0.0
    else:
        beta2 = 1 - M**2
        D     = np.sqrt(1 - beta2*d**2)
        ath_D = np.arctanh(D)
        a     = 2*beta2*d**2 / D**3 * (ath_D - D)
        du    = a / (2-a) / np.sqrt(beta2)
    
    C  = 2.3
    ff = (1 + C*du)**2
    return ff



if __name__ == '__main__':
    print 'RUNNING TEST'
    
    fineness = 6.0
    Mach     = 0.5
    
    #Method Test
    print '<<Test run of the body_form_factor() method>> \n'
    print 'Body at Mach {}'.format(Mach)
    print 'Fineness ratio: {}'.format(fineness)
    
    form_factor = body_form_factor(fineness, Mach)
    #Compare to AA241 Notes - http://adg.stanford.edu/aa241/AircraftDesign.html    
    expected    = 1.203
    
    print ' Form factor    = {0:.3f}m'.format(form_factor)
    print ' Expected value = {}m'.format(expected)
    print ' Percent Error  = {0:.2f}%'.format(100.0*(form_factor-expected)/expected)
    