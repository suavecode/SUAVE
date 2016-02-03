# Aerodynamics.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data, Data_Exception, Data_Warning
from SUAVE.Analyses import Analysis

# default Aero Results
from Results import Results

import numpy as np

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

class Aerodynamics(Analysis):
    """ SUAVE.Analyses.Aerodynamics.Aerodynamics()
    """
    def __defaults__(self):
        self.tag    = 'aerodynamics'
        
        self.geometry = Data()
        self.settings = Data()
        self.settings.maximum_lift_coefficient = np.inf
        
        
    def evaluate(self,state):
        
        results = Results()
        
        return results
    
    def finalize(self):
        
        return     
    
    
    def compute_forces(self,conditions):
        
        # unpack
        q    = conditions.freestream.dynamic_pressure
        Sref = self.geometry.reference_area
        
        # 
        CL = conditions.aerodynamics.lift_coefficient
        CD = conditions.aerodynamics.drag_coefficient
        
        N = q.shape[0]
        L = np.zeros([N,3])
        D = np.zeros([N,3])

        L[:,2] = ( -CL * q * Sref )[:,0]
        D[:,0] = ( -CD * q * Sref )[:,0]

        results = Data()
        results.lift_force_vector = L
        results.drag_force_vector = D

        return results        
    
        
        