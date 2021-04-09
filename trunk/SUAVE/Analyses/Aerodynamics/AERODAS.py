## @ingroup Analyses-Aerodynamics
# AERODAS.py
# 
# Created:  Feb 2016, E. Botero
# Modified: Apr 2019, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Analyses.Aerodynamics.Markup import Markup
from SUAVE.Core import Data, Units
from SUAVE.Analyses import Process
from SUAVE.Analyses.Aerodynamics.Process_Geometry import Process_Geometry
from SUAVE.Methods.Aerodynamics import AERODAS as Methods
from SUAVE.Methods.Aerodynamics.Common import Fidelity_Zero as Common

# ----------------------------------------------------------------------
#  AERODAS
# ----------------------------------------------------------------------
## @ingroup Analyses-Aerodynamics
class AERODAS(Markup):
    """This is an analysis based on the AERODAS models.
    
    Assumptions:
    None
    
    Source:
    NASA TR: "Models of Lift and Drag Coefficients of Stalled and Unstalled Airfoils in
      Wind Turbines and Wind Tunnels" by D. A. Spera
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
        self.tag = 'AERODAS Model'
        
        settings = self.settings
        settings.section_zero_lift_angle_of_attack                = 0.0 * Units.deg
        settings.section_minimum_drag_coefficient_angle_of_attack = 0.0 * Units.deg 
        settings.section_lift_curve_slope                         = 2.0 * np.pi

        # build the evaluation process
        compute = self.process.compute
        
        compute.setup_data = Methods.AERODAS_setup.setup_data
    
        # Get all of the coefficients for AERODAS wings
        compute.wings_coefficients = Process()
        compute.wings_coefficients = Process_Geometry('wings')
        compute.wings_coefficients.section_properties  = Methods.section_properties.section_properties
        compute.wings_coefficients.finite_aspect_ratio = Methods.finite_aspect_ratio.finite_aspect_ratio
        compute.wings_coefficients.pre_stall           = Methods.pre_stall_coefficients.pre_stall_coefficients
        compute.wings_coefficients.post_stall          = Methods.post_stall_coefficients.post_stall_coefficients
        
        # Fuselage drag?
        # do a plate build up with angles
        
        # Miscellaneous drag?
        # Compressibility corrections?
        
        compute.lift_drag_total                        = Methods.AERODAS_setup.lift_drag_total
        
        compute.lift = Process()
        compute.lift.total                             = Common.Lift.aircraft_total
        compute.drag = Process()
        compute.drag.total                             = Methods.AERODAS_setup.drag_total
        
        def initialize(self):
            super(AERODAS, self).initialize()
            self.process.compute.lift.inviscid_wings.geometry = self.geometry
            self.process.compute.lift.inviscid_wings.initialize()
            
        finalize = initialize   