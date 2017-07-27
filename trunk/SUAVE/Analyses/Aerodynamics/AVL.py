# AVL.py
#
# Created: Apr 2017, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data
from Markup import Markup
from SUAVE.Analyses import Process
import numpy as np

# Default aero Results
from Results import Results

# The aero methods
from SUAVE.Methods.Aerodynamics import Fidelity_Zero as Methods
from Process_Geometry import Process_Geometry
from SUAVE.Analyses.Aerodynamics.AVL_Inviscid import AVL_Inviscid

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------
class AVL(Markup):
    
    def __defaults__(self):
        
        self.tag    = 'AVL_markup'       
    
        # Correction factors
        settings = self.settings
        settings.trim_drag_correction_factor        = 1.02
        settings.wing_parasite_drag_form_factor     = 1.1
        settings.fuselage_parasite_drag_form_factor = 2.3
        settings.oswald_efficiency_factor           = None
        settings.viscous_lift_dependent_drag_factor = 0.38
        settings.drag_coefficient_increment         = 0.0000
        settings.spoiler_drag_increment             = 0.00 
        settings.maximum_lift_coefficient           = np.inf 
        
                
        # Build the evaluation process
        compute = self.process.compute
        compute.lift = Process()

        # Run AVL to determine lift
        compute.lift.inviscid                         = AVL_Inviscid()
        compute.lift.total                            = SUAVE.Methods.Aerodynamics.AERODAS.AERODAS_setup.lift_total
        
        # Do a traditional drag buildup
        compute.drag = Process()
        compute.drag.parasite                      = Process()
        compute.drag.parasite.wings                = Process_Geometry('wings')
        compute.drag.parasite.wings.wing           = Methods.Drag.parasite_drag_wing 
        compute.drag.parasite.fuselages            = Process_Geometry('fuselages')
        compute.drag.parasite.fuselages.fuselage   = Methods.Drag.parasite_drag_fuselage
        compute.drag.parasite.propulsors           = Process_Geometry('propulsors')
        compute.drag.parasite.propulsors.propulsor = Methods.Drag.parasite_drag_propulsor
        compute.drag.parasite.pylons               = Methods.Drag.parasite_drag_pylon
        compute.drag.parasite.total                = Methods.Drag.parasite_total
        compute.drag.induced                       = Methods.Drag.induced_drag_aircraft
        compute.drag.compressibility               = Process()
        compute.drag.compressibility.wings         = Process_Geometry('wings')
        compute.drag.compressibility.wings.wing    = Methods.Drag.compressibility_drag_wing
        compute.drag.compressibility.total         = Methods.Drag.compressibility_drag_wing_total        
        compute.drag.miscellaneous                 = Methods.Drag.miscellaneous_drag_aircraft_ESDU
        compute.drag.untrimmed                     = SUAVE.Methods.Aerodynamics.AVL.untrimmed
        compute.drag.trim                          = Methods.Drag.trim
        compute.drag.spoiler                       = Methods.Drag.spoiler_drag
        compute.drag.total                         = SUAVE.Methods.Aerodynamics.SU2_Euler.total_aircraft_drag
        
        
    def initialize(self):
        self.process.compute.lift.inviscid.geometry = self.geometry
        
        # Generate the surrogate
        self.process.compute.lift.inviscid.initialize()
        
    finalize = initialize
    
    

