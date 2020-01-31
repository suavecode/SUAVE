## @ingroup Analyses-Aerodynamics
# Aerodynamics.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
from SUAVE.Analyses import Analysis

# default Aero Results
from .Results import Results

import numpy as np

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------
## @ingroup Analyses-Aerodynamics
class Aerodynamics(Analysis):
    """This is the base class for aerodynamics analyses. It contains functions
    that are built into the default class.
    
    Assumptions:
    None
    
    Source:
    N/A
    """
    def __defaults__(self):
        """This sets the default values and methods for the analysis.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        N/A
        """           
        self.tag    = 'aerodynamics'
        
        self.geometry = Data()
        self.settings = Data()
        self.settings.maximum_lift_coefficient = np.inf
        
        
    def evaluate(self,state):
        """The default evaluate function.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        results   <Results class> (empty)

        Properties Used:
        N/A
        """           
        
        results = Data()
        
        return results
    
    def finalize(self):
        """The default finalize function.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        N/A
        """         
        
        return     
    
    
    def compute_forces(self,conditions):
        """The default function to compute forces.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        conditions.freestream.
          dynamic_pressure       [Pa]
        conditions.aerodynamics.
          lift_coefficient       [-]
          drag_coefficient       [-]

        Outputs:
        results.
          lift_force_vector      [N]
          drag_force_vector      [N]

        Properties Used:
        self.geometry.reference_area [m^2]
        """          
        
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
    
        
        