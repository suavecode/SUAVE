# SU2_Euler.py
#
# Created:  Sep 2016, E. Botero
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data
from Markup import Markup
from SUAVE.Analyses import Process
import numpy as np

# default Aero Results
from Results import Results

# the aero methods
from SUAVE.Methods.Aerodynamics import Fidelity_Zero as Methods
from Process_Geometry import Process_Geometry
from Vortex_Lattice import Vortex_Lattice

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------
class SU2_Euler(Markup):
    
    def __defaults__(self):
        
        self.tag    = 'SU2_Euler_markup'       
    
        # correction factors
        settings = self.settings
        settings.trim_drag_correction_factor        = 1.02
        settings.wing_parasite_drag_form_factor     = 1.1
        settings.fuselage_parasite_drag_form_factor = 2.3
        settings.oswald_efficiency_factor           = None
        settings.viscous_lift_dependent_drag_factor = 0.38
        settings.drag_coefficient_increment         = 0.0000
        settings.spoiler_drag_increment             = 0.00 
        settings.maximum_lift_coefficient           = np.inf 
        
        # build the evaluation process
        compute = self.process.compute
        compute.lift = Process()
        
        # Some stuff for meshing

        # Run SU2
        compute.lift.inviscid                      = SU2_inviscid()
        
        # Do a traditional drag buildup for viscous components
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
        compute.drag.miscellaneous                 = Methods.Drag.miscellaneous_drag_aircraft_ESDU
        compute.drag.untrimmed                     = SUAVE.Methods.Aerodynamics.SU2_Euler.untrimmed
        compute.drag.trim                          = Methods.Drag.trim
        compute.drag.spoiler                       = Methods.Drag.spoiler_drag
        compute.drag.total                         = SUAVE.Methods.Aerodynamics.SU2_Euler.total_aircraft_drag
        
        
    def initialize(self):
        # Mesh the Geometry
        self.process.compute.lift.inviscid.geometry = self.geometry
        
        # Generate the surrogate
        self.process.compute.lift.inviscid.initialize()
        
    finalize = initialize