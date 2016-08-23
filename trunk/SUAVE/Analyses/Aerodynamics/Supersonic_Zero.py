# Supersonic_Zero.py
# 
# Created:  Tim MacDonald, based on Fidelity_Zero
# Modified: Tim MacDonald, 1/29/15 
# Modified: Feb 2016, Andrew Wendorff
#
# Updated for new optimization structure


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data
from Markup import Markup
from SUAVE.Analyses import Process

from Vortex_Lattice import Vortex_Lattice
from Process_Geometry import Process_Geometry
from SUAVE.Methods.Aerodynamics import Supersonic_Zero as Methods

import numpy as np
#from SUAVE.Attributes.Aerodynamics.Aerodynamics_1d_Surrogate import Aerodynamics_1d_Surrogate

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Supersonic_Zero(Markup):
    """ SUAVE.Attributes.Aerodynamics.Fidelity_Zero
        aerodynamic model that builds a surrogate model for clean wing 
        lift, using vortex lattic, and various handbook methods
        for everything else
        
        this class is callable, see self.__call__
        
    """
    
    def __defaults__(self):
        
        self.tag = 'Fidelity_Zero_Supersonic'
        
        # correction factors
        settings =  self.settings
        settings.fuselage_lift_correction           = 1.14
        settings.trim_drag_correction_factor        = 1.02
        settings.wing_parasite_drag_form_factor     = 1.1
        settings.fuselage_parasite_drag_form_factor = 2.3
        settings.aircraft_span_efficiency_factor    = 0.78
        settings.drag_coefficient_increment         = 0.0000
        settings.maximum_lift_coefficient           = np.inf 
        
        # vortex lattice configurations
        settings.number_panels_spanwise = 5
        settings.number_panels_chordwise = 1
        
        
        # build the evaluation process
        compute = self.process.compute
        
        compute.lift = Process()
        compute.lift.inviscid_wings                = Vortex_Lattice()
        compute.lift.vortex                        = Methods.Lift.vortex_lift  # SZ
        compute.lift.compressible_wings            = Methods.Lift.wing_compressibility # SZ
        compute.lift.fuselage                      = Methods.Lift.fuselage_correction # difference in results storage
        compute.lift.total                         = Methods.Lift.aircraft_total # no difference
        
        compute.drag = Process()
        compute.drag.parasite                      = Process()
        compute.drag.parasite.wings                = Process_Geometry('wings')
        compute.drag.parasite.wings.wing           = Methods.Drag.parasite_drag_wing # SZ
        compute.drag.parasite.fuselages            = Process_Geometry('fuselages')
        compute.drag.parasite.fuselages.fuselage   = Methods.Drag.parasite_drag_fuselage # SZ
        compute.drag.parasite.propulsors           = Process_Geometry('propulsors')
        compute.drag.parasite.propulsors.propulsor = Methods.Drag.parasite_drag_propulsor # SZ
        #compute.drag.parasite.pylons               = Methods.Drag.parasite_drag_pylon
        compute.drag.parasite.total                = Methods.Drag.parasite_total # SZ
        compute.drag.induced                       = Methods.Drag.induced_drag_aircraft # SZ
        compute.drag.compressibility               = Process()
        compute.drag.compressibility.total         = Methods.Drag.compressibility_drag_total # SZ
        compute.drag.miscellaneous                 = Methods.Drag.miscellaneous_drag_aircraft # different type used in FZ
        compute.drag.untrimmed                     = Methods.Drag.untrimmed # SZ can be changed to match
        compute.drag.trim                          = Methods.Drag.trim # SZ can be chanaged to match
        compute.drag.total                         = Methods.Drag.total_aircraft # SZ
        
        
    def initialize(self):
        self.process.compute.lift.inviscid_wings.geometry = self.geometry
        self.process.compute.lift.inviscid_wings.initialize()
        
    finalize = initialize        
