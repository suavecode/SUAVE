## @ingroup Analyses-Aerodynamics
# Fidelity_Zero.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff
#           Apr 2019, T. MacDonald
#           Apr 2020, M. Clarke
#           Sep 2020, M. Clarke 
#           Jun 2021, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data
from .Markup import Markup
from SUAVE.Analyses import Process
import numpy as np

# the aero methods
from SUAVE.Methods.Aerodynamics import Fidelity_Zero as Methods
from SUAVE.Methods.Aerodynamics.Common import Fidelity_Zero as Common
from .Process_Geometry import Process_Geometry
from .Vortex_Lattice import Vortex_Lattice

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------
## @ingroup Analyses-Aerodynamics
class Fidelity_Zero(Markup):
    """This is an analysis based on low-fidelity models.

    Assumptions:
    Subsonic

    Source:
    Primarily based on adg.stanford.edu, see methods for details
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
        self.tag    = 'fidelity_zero_markup'
    
        # correction factors
        settings = self.settings
        settings.fuselage_lift_correction           = 1.14
        settings.trim_drag_correction_factor        = 1.02
        settings.wing_parasite_drag_form_factor     = 1.1
        settings.fuselage_parasite_drag_form_factor = 2.3
        settings.oswald_efficiency_factor           = None
        settings.span_efficiency                    = None
        settings.viscous_lift_dependent_drag_factor = 0.38
        settings.drag_coefficient_increment         = 0.0000
        settings.spoiler_drag_increment             = 0.00 
        settings.maximum_lift_coefficient           = np.inf
        settings.number_spanwise_vortices           = None 
        settings.number_chordwise_vortices          = None 
        settings.initial_timestep_offset            = 0.
        settings.wake_development_time              = 0.05
        settings.number_of_wake_timesteps           = 30
        settings.use_surrogate                      = True
        settings.propeller_wake_model               = False 
        settings.use_bemt_wake_model                = False 
        settings.discretize_control_surfaces        = False
        settings.model_fuselage                     = False
        settings.recalculate_total_wetted_area      = False
        settings.model_nacelle                      = False

        # build the evaluation process
        compute = self.process.compute
        
        compute.lift = Process()

        compute.lift.inviscid_wings                = Vortex_Lattice()
        compute.lift.vortex                        = SUAVE.Methods.skip
        compute.lift.fuselage                      = Common.Lift.fuselage_correction
        compute.lift.total                         = Common.Lift.aircraft_total
        
        compute.drag = Process()
        compute.drag.parasite                      = Process()
        compute.drag.parasite.wings                = Process_Geometry('wings')
        compute.drag.parasite.wings.wing           = Common.Drag.parasite_drag_wing 
        compute.drag.parasite.fuselages            = Process_Geometry('fuselages')
        compute.drag.parasite.fuselages.fuselage   = Common.Drag.parasite_drag_fuselage
        compute.drag.parasite.nacelles             = Process_Geometry('nacelles')
        compute.drag.parasite.nacelles.nacelle     = Common.Drag.parasite_drag_nacelle
        compute.drag.parasite.pylons               = Common.Drag.parasite_drag_pylon
        compute.drag.parasite.total                = Common.Drag.parasite_total
        compute.drag.induced                       = Common.Drag.induced_drag_aircraft
        compute.drag.compressibility               = Process()
        compute.drag.compressibility.wings         = Process_Geometry('wings')
        compute.drag.compressibility.wings.wing    = Common.Drag.compressibility_drag_wing
        compute.drag.compressibility.total         = Common.Drag.compressibility_drag_wing_total
        compute.drag.miscellaneous                 = Common.Drag.miscellaneous_drag_aircraft_ESDU
        compute.drag.untrimmed                     = Common.Drag.untrimmed
        compute.drag.trim                          = Common.Drag.trim
        compute.drag.spoiler                       = Common.Drag.spoiler_drag
        compute.drag.total                         = Common.Drag.total_aircraft
        
        # Set subsonic mach numbers for the vortex lattice surrogate
        compute.lift.inviscid_wings.training.Mach = np.array([[0.0, 0.1  , 0.2 , 0.3,  0.5,  0.75 , 0.85 , 0.9]]).T     
        
        
    def initialize(self):
        """Initializes the surrogate needed for lift calculation.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        self.geometry
        """                  
        super(Fidelity_Zero, self).initialize()
        
        use_surrogate             = self.settings.use_surrogate
        propeller_wake_model      = self.settings.propeller_wake_model 
        use_bemt_wake_model       = self.settings.use_bemt_wake_model
        n_sw                      = self.settings.number_spanwise_vortices
        n_cw                      = self.settings.number_chordwise_vortices
        ito                       = self.settings.initial_timestep_offset
        wdt                       = self.settings.wake_development_time
        nwts                      = self.settings.number_of_wake_timesteps
        mf                        = self.settings.model_fuselage
        mn                        = self.settings.model_nacelle
        dcs                       = self.settings.discretize_control_surfaces

        self.process.compute.lift.inviscid_wings.geometry = self.geometry 
        self.process.compute.lift.inviscid_wings.initialize(use_surrogate,n_sw,n_cw,propeller_wake_model,use_bemt_wake_model,ito,wdt,nwts,mf,mn,dcs )
                                                            
    finalize = initialize                                          