""" External_Data.py: Aerodynamic model based on user-supplied data """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Attributes.Aerodynamics import Aerodynamics

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------
    
class External_Data(Aerodynamics):

    """ SUAVE.Attributes.Aerodyanmics.ExternalData: Aerodynamic model based on user-supplied data """
    
    def __defaults__(self):                             # can handle CD(alpha), CL(alpha); needs Mach support (spline surface)
        self.tag = 'External Data'
        self.S = 1.0                                    # reference area (m^2)
        self.M = []                                     # Mach vector
        self.alpha = []                                 # alpha vector
        self.CD = []                                    # CD vector
        self.CL = []                                    # CL vector

    def initialize(alpha,CD,CL):

        #create_fit(alpha,CL)
        #create_fit(alpha,CD)

        #return
        raise NotImplementedError

    def __call__(self,alpha,segment):

        #CL = evaluate_fit(alpha)
        #CD = evaluate_fit(alpha)

        #return CD, CL
        raise NotImplementedError
