## @ingroup Components-Energy-Distributors
# Cryogenic_Lead.py
#
# Created:  Feb 2020, K.Hamilton

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  Cryogenic Lead Class
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Distributors
class Cryogenic_Lead(Energy_Component):
    
    def __defaults__(self):
        """ This sets the default values.
    
            Assumptions:
            Cryogenic Leads only operate at their optimum current, or at zero current.
    
            Source:
            Current Lead Optimization for Cryogenic Operation at Intermediate Temperatures - Broomberg
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
            """         
        
        self.cold_temp          =   77.0    # [K]
        self.hot_temp           =  300.0    # [K]
        self.current            = 1000.0    # [A]
        self.length             =    1.0    # [m]
    
        self.cross_section      =    1.0    # [m2]
        self.optimum_current    = 1000.0    # [A]
        self.minimum_Q          = 1000.0    # [W]
        self.unpowered_Q        = 1000.0    # [W]
