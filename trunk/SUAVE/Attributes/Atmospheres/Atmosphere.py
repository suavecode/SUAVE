""" Atmosphere.py: Constant-property atmopshere model """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# classes
import numpy as np
from SUAVE.Attributes.Gases import Air
from SUAVE.Attributes.Constants import Constant, Composition
from SUAVE.Structure import Data, Data_Exception, Data_Warning

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
        self.pressure              = 0.0      # Pa
        self.temperature           = 0.0      # K
        self.density               = 0.0      # kg/m^3
        self.speed_of_sound        = 0.0      # m/s
        self.dynamic_viscosity     = 0.0      # Pa-s
        self.composition           = Data()
        self.composition.gas       = 1.0

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
            return (self.pressure*O, self.temperature*O, self.density*O, self.speed_of_sound*O, self.dynamic_viscosity*O)

        if type.lower() in pressure:
            return self.pressure*O
                      
        elif type.lower() in temp:
            return self.temperature*O

        elif type.lower() in density:
            return self.density*O
            
        elif type.lower() in speedofsound:
            return self.speed_of_sound*O

        elif type.lower() in viscosity:
            return self.dynamic_viscosity*O

        else:
            print "Unknown data, " + type + ", request; no data returned."
            return []
    
    