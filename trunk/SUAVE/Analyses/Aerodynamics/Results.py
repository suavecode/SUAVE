## @ingroup Analyses-Aerodynamics
# Results
# Created:   Trent, Jan 2014
# Modified:  Andrew Wendorff, Feb 2016

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Analyses import Results as Base_Results
import numpy as np

# ----------------------------------------------------------------------
#  Default Aerodynamic Results
# ----------------------------------------------------------------------


default_result = np.zeros([1,1])
## @ingroup Analyses-Aerodynamics
class Results(Base_Results):
    """A 

    Assumptions:
    Subsonic

    Source:
    Primarily based on adg.stanford.edu, see methods for details
    """      
    def __defaults__(self):
        
        self.lift_coefficient  = default_result * 0.
        self.drag_coefficient  = default_result * 0.
        
        self.lift_force_vector = default_result * 0.
        self.drag_force_vector = default_result * 0.
