""" Atmosphere.py: Constant-property atmopshere model """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# classes
import numpy as np
from SUAVE.Attributes.Gases import Air
from SUAVE.Attributes.Constants import Constant, Composition

# other
from numpy import sqrt, exp, abs

# ----------------------------------------------------------------------
#  Atmosphere Data Class
# ----------------------------------------------------------------------

class Atmosphere(Constant):

    """ SUAVE.Attributes.Atmospheres.Atmosphere
    """

    def __defaults__(self):
        self.tag = 'Constant-property atmopshere'
        self.p      = 0.0      # Pa
        self.T      = 0.0      # K
        self.rho    = 0.0      # kg/m^3
        self.a      = 0.0      # m/s
        self.mew    = 0.0      # Pa-s
        self.Composition = Composition( Gas = 1.0 )

    def compute_values(self,altitude,type="all"):
    
        # return options
        all_vars = ["all", "everything"]
        pressure = ["p", "pressure"]
        temp = ["t", "temp", "temperature"]
        density = ["rho", "density"]
        speedofsound = ["speedofsound", "a"]
        viscosity = ["viscosity", "mew"]

        # convert input if necessary
        if isinstance(altitude, int): 
            altitude = np.array([float(altitude)])
        elif isinstance(altitude, float):
            altitude = np.array([altitude])

        O = np.ones(len(altitude))

        # return requested data
        if type.lower() in all_vars:
            return (self.p*O, self.T*O, self.rho*O, self.a*O, self.mew*O)

        if type.lower() in pressure:
            return self.p*O
                      
        elif type.lower() in temp:
            return self.T*O

        elif type.lower() in density:
            return self.rho*O
            
        elif type.lower() in speedofsound:
            return self.a*O

        elif type.lower() in viscosity:
            return self.mew*O

        else:
            print "Unknown data, " + type + ", request; no data returned."
            return []
    
    