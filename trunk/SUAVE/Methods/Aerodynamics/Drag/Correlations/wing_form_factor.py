# wing_form_factor.py
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

def wing_form_factor(wing,Mach):
    """ ff = SUAVE.Methods.Aerodynamics.Drag.Correlations.wing_form_factor(wing,Mach)
        This method computes the parasitic drag form factor of an aerodynamic
        surface
        
        Inputs:
            wing - a data dictionary with the fields:
                t_c - the average thickness-to-chord of the wing [dimensionless]
                sweep - the wing sweep [radians]
            Mach - the flight Mach number at which form factor is evaluated
                
        Outputs:
            ff - the form factor of the aerodynamic surface
                           
        Assumptions:
            - Assumes the reduction in friction coefficient due to Reynolds number
            and local Mach number is neglible
            - Uses Professor Ilan Kroo's correlation from the AA241 Course Notes.
    """             
    # Unpack inputs
    sweep = wing.sweep
    t_c   = wing.t_c
    M     = Mach
    
    #Compute form factor
    cosL = np.cos(sweep)
    if M*cosL > 1.0:
        ff = 1.0
    else:
        C     = 1.1
        beta2 = (1.0 -(M*cosL)**2)
        k1    = 2*C * cosL**2 * t_c / np.sqrt(beta2)
        k2    = C**2 * cosL**2 * t_c**2 * (1+5*cosL**2) / (2*beta2)
        ff    = 1 + k1 + k2
    
    return ff



if __name__ == '__main__':
    print 'RUNNING TEST'
    
    wing       = Data()
    wing.sweep = 25 * Units.deg
    wing.t_c   = 0.1
    
    Mach       = 0.75
    
    #Method Test
    print '<<Test run of the wing_form_factor() method>> \n'
    print 'Wing at Mach {}'.format(Mach)
    print 't/c = {0},  sweep = {1} degrees'.format(wing.t_c,wing.sweep/Units.deg)
    
    form_factor = wing_form_factor(wing, Mach)
    #Compare to AA241 Notes - http://adg.stanford.edu/aa241/AircraftDesign.html    
    expected    = 1.294
    
    print ' Form factor    = {0:.3f}m'.format(form_factor)
    print ' Expected value = {}m'.format(expected)
    print ' Percent Error  = {0:.2f}%'.format(100.0*(form_factor-expected)/expected)
    