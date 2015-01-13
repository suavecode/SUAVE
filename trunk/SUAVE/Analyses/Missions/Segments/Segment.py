
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# python imports
import numpy as np
from copy import deepcopy

# SUAVE imports
from Base_Segment                       import Base_Segment
from SUAVE.Core                    import Data, Data_Exception
from SUAVE.Core                    import Container as ContainerBase
from SUAVE.Methods.Utilities.Chebyshev  import chebyshev_data
from SUAVE.Methods.Utilities            import atleast_2d_col
from SUAVE.Analyses                     import Analysis

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

class Segment(Data):
    
    # ------------------------------------------------------------------
    #   Data Defaults
    # ------------------------------------------------------------------    

    def __defaults__(self):
        self.tag = 'segment'
        
        # --------------------------------------------------------------
        #  Strategy 
        self.strategy = Data()
        
        # --------------------------------------------------------------
        #  Analyses 
        self.analyses = Analysis.Container()
        
        # --------------------------------------------------------------
        #  Conditions 
        self.conditions = Data()
        
        # --------------------------------------------------------------
        #  Unknowns 
        self.unknowns = Data()
        
        # --------------------------------------------------------------
        #  Residuals
        self.residuals = Data()
        
        return

# ----------------------------------------------------------------------
#  Handle Linking
# ----------------------------------------------------------------------

class Container(ContainerBase):
    pass

Segment.Container = Container


