""" Propulsion.py: Methods for Propulsion Analysis """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Data
from SUAVE.Attributes import Constants
# import SUAVE

# ----------------------------------------------------------------------
#  Mission Methods
# ----------------------------------------------------------------------
   
def Cp(T):

    #gamma=1.4
    Thetav=3354
    R=287
    Cpt=R*(7/2+((Thetav/(2*T))/(np.sinh(Thetav/(2*T))))**2)
    
    #Cpt = 1.9327e-10*T**4 - 7.9999e-7*T**3 + 1.1407e-3*T**2 - 4.4890e-1*T + 1.0575e+3
    return Cpt