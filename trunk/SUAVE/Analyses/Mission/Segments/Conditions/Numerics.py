## @ingroup Analyses-Mission-Segments-Conditions
# Numerics.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from .Conditions import Conditions

from SUAVE.Methods.Utilities.Chebyshev  import chebyshev_data

import numpy as np

# ----------------------------------------------------------------------
#  Numerics
# ----------------------------------------------------------------------

## @ingroup Analyses-Mission-Segments-Conditions
class Numerics(Conditions):
    """ Creates the data structure for the numerical solving of a mission.
    
        Assumptions:
        None
        
        Source:
        None
    """
    
    def __defaults__(self):
        """This sets the default values.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
        """           
        self.tag = 'numerics'
        
        self.number_control_points = 16
        self.discretization_method = chebyshev_data
        
        self.solver_jacobian                  = "none"
        self.tolerance_solution               = 1e-8
        self.converged                        = None
        self.max_evaluations                  = 0.
        self.step_size                        = None
        
        self.dimensionless = Conditions()
        self.dimensionless.control_points = np.empty([0,0])
        self.dimensionless.differentiate  = np.empty([0,0])
        self.dimensionless.integrate      = np.empty([0,0]) 
        
        self.time = Conditions()
        self.time.control_points = np.empty([0,0])
        self.time.differentiate  = np.empty([0,0])
        self.time.integrate      = np.empty([0,0]) 
        
        
        
        