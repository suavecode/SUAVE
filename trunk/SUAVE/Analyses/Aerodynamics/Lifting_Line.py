## @ingroup Analyses-Aerodynamics
# Lifting_Line.py
# 
# Created:  Aug 2017, E. Botero
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Analyses.Aerodynamics.Markup import Markup
from SUAVE.Core import Data, Units
from SUAVE.Analyses import Process
from SUAVE.Analyses.Aerodynamics.Process_Geometry import Process_Geometry
from SUAVE.Methods.Aerodynamics import Lifting_Line as Methods

# ----------------------------------------------------------------------
#  AERODAS
# ----------------------------------------------------------------------
## @ingroup Analyses-Aerodynamics
class Lifting_Line(Markup):
    """This is an analysis based on an extended lifting line for wings
    and fidelity zero methods for other sources of drag.
    
    Assumptions:
    None
    
    Source:
    Traub, L. W., Botero, E., Waghela, R., Callahan, R., & Watson, A. (2015). Effect of Taper Ratio at Low Reynolds Number. Journal of Aircraft.
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
        self.tag = 'Lifting_Line Model'
        
        settings = self.settings

        # build the evaluation process
        compute = self.process.compute
        
    
        # Get all of the coefficients for AERODAS wings

        
        # Fuselage drag?
        
        # Miscellaneous drag?

        compute.lift_drag_total                        = Methods.AERODAS_setup.lift_drag_total
        
        compute.lift = Process()
        compute.lift.total                             = Methods.AERODAS_setup.lift_total
        compute.drag = Process()
        compute.drag.total                             = Methods.AERODAS_setup.drag_total
        
        def initialize(self):
            self.process.compute.lift.inviscid_wings.geometry = self.geometry
            self.process.compute.lift.inviscid_wings.initialize()
            
        finalize = initialize   