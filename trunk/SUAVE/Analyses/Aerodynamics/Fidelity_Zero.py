
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data, Data_Exception, Data_Warning
from Markup import Markup
from SUAVE.Analyses import Process

# default Aero Results
from Results import Results

# the aero methods
from SUAVE.Methods.Aerodynamics import Fidelity_Zero as Methods
from Process_Geometry import Process_Geometry
#from SUAVE.Analyses.Aerodynamics import Inviscid_Wings_Lift

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------
class Fidelity_Zero(Markup):
    
    def __defaults__(self):
        
        self.tag    = 'fidelity_zero_markup'
        
        ## available from Markup
        #self.geometry = Data()
        #self.settings = Data()
        
        #self.process = Process()
        #self.process.initialize = Process()
        #self.process.compute = Process()        
    
        # correction factors
        settings = self.settings
        settings.fuselage_lift_correction           = 1.14
        settings.trim_drag_correction_factor        = 1.02
        settings.wing_parasite_drag_form_factor     = 1.1
        settings.fuselage_parasite_drag_form_factor = 2.3
        settings.aircraft_span_efficiency_factor    = 0.78
        settings.drag_coefficient_increment         = 0.0000
        settings.wing_span_efficiency               = 0.90
        
        # vortex lattice configurations
        settings.number_panels_spanwise  = 5
        settings.number_panels_chordwise = 1
        
        
        # build the evaluation process
        compute = self.process.compute
        
        # these methods have interface as
        # results = function(state,settings,geometry)
        # results are optional
        
        # first stub out empty functions
        # then implement methods
        # then we'll figure out how to connect to a mission
        
        compute.lift = Process()
        #compute.lift.inviscid_wings                = Inviscid_Wings_Lift
        compute.lift.vortex                        = SUAVE.Methods.skip
        #compute.lift.compressible_wings            = Methods.Lift.wing_compressibilty_correction
        compute.lift.fuselage                      = Methods.Lift.fuselage_correction
        compute.lift.total                         = Methods.Lift.aircraft_total
        
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
        #compute.drag.compressibility.wings         = Process_Geometry('wings')
        #compute.drag.compressibility.wings         = Methods.Drag.compressibiltiy_drag_wing
        #compute.drag.compressibility.wings.wing    = Methods.Drag.compressibiltiy_drag_wing
        compute.drag.miscellaneous                 = Methods.Drag.miscellaneous_drag_aircraft_ESDU
        compute.drag.untrimmed                     = Methods.Drag.untrimmed
        compute.drag.trim                          = Methods.Drag.trim
        compute.drag.total                         = Methods.Drag.total_aircraft
        
        
        