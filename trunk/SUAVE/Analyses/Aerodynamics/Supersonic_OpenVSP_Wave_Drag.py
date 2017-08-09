## @ingroup Analyses-Aerodynamics
# Supersonic_OpenVSP_Wave_Drag.py
# 
# Created:            T. MacDonald
# Modified: Apr 2017, T. MacDonald


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
from SUAVE.Methods.Aerodynamics import OpenVSP_Wave_Drag as VSP_Methods

import numpy as np

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------
## @ingroup Analyses-Aerodynamics
class Supersonic_OpenVSP_Wave_Drag(Markup):

    
    def __defaults__(self):
        
        self.tag = 'Supersonic_OpenVSP_Wave_Drag'
        
        # correction factors
        settings =  self.settings
        settings.fuselage_lift_correction           = 1.14
        settings.trim_drag_correction_factor        = 1.02
        settings.wing_parasite_drag_form_factor     = 1.1
        settings.fuselage_parasite_drag_form_factor = 2.3
        settings.aircraft_span_efficiency_factor    = 0.78
        settings.viscous_lift_dependent_drag_factor = 0.38
        settings.drag_coefficient_increment         = 0.0000
        settings.oswald_efficiency_factor           = None
        settings.maximum_lift_coefficient           = np.inf
        settings.number_slices                      = 20
        settings.number_rotations                   = 10
        
        # vortex lattice configurations
        settings.number_panels_spanwise = 5
        settings.number_panels_chordwise = 1
        
        
        # build the evaluation process
        compute = self.process.compute
        
        compute.lift = Process()
        compute.lift.inviscid_wings                = Vortex_Lattice()
        compute.lift.vortex                        = Methods.Lift.vortex_lift
        compute.lift.compressible_wings            = Methods.Lift.wing_compressibility
        compute.lift.fuselage                      = Methods.Lift.fuselage_correction
        compute.lift.total                         = Methods.Lift.aircraft_total
        
        compute.drag = Process()
        compute.drag.compressibility               = Process()
        compute.drag.compressibility.total         = VSP_Methods.compressibility_drag_total       
        compute.drag.parasite                      = Process()
        compute.drag.parasite.wings                = Process_Geometry('wings')
        compute.drag.parasite.wings.wing           = Methods.Drag.parasite_drag_wing
        compute.drag.parasite.fuselages            = Process_Geometry('fuselages')
        compute.drag.parasite.fuselages.fuselage   = Methods.Drag.parasite_drag_fuselage
        compute.drag.parasite.propulsors           = Process_Geometry('propulsors')
        compute.drag.parasite.propulsors.propulsor = Methods.Drag.parasite_drag_propulsor
        #compute.drag.parasite.pylons               = Methods.Drag.parasite_drag_pylon # supersonic pylon methods not currently available
        compute.drag.parasite.total                = Methods.Drag.parasite_total
        compute.drag.induced                       = Methods.Drag.induced_drag_aircraft
        compute.drag.miscellaneous                 = Methods.Drag.miscellaneous_drag_aircraft
        compute.drag.untrimmed                     = Methods.Drag.untrimmed 
        compute.drag.trim                          = Methods.Drag.trim
        compute.drag.total                         = Methods.Drag.total_aircraft
        
        
    def initialize(self):
        import os
        
        # Remove old volume drag data so that new data can be appended without issues
        try:
            os.remove('volume_drag_data_' + self.geometry.tag + '.npy')  
        except:
            pass
        
        self.process.compute.lift.inviscid_wings.geometry = self.geometry
        self.process.compute.lift.inviscid_wings.initialize()
        
    finalize = initialize        