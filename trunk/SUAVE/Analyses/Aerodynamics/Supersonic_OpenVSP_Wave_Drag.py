## @ingroup Analyses-Aerodynamics
# Supersonic_OpenVSP_Wave_Drag.py
# 
# Created:            T. MacDonald
# Modified: Apr 2017, T. MacDonald
#           Apr 2019, T. MacDonald
#           Apr 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data
from .Markup import Markup
from SUAVE.Analyses import Process

from .Vortex_Lattice import Vortex_Lattice
from .Process_Geometry import Process_Geometry
from SUAVE.Methods.Aerodynamics import Supersonic_Zero  as Methods
from SUAVE.Methods.Aerodynamics import OpenVSP_Wave_Drag as VSP_Methods
from SUAVE.Methods.Aerodynamics.Common import Fidelity_Zero as Common

import numpy as np

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------
## @ingroup Analyses-Aerodynamics
class Supersonic_OpenVSP_Wave_Drag(Markup):
    """This is an analysis based on low-fidelity models.

    Assumptions:
    None

    Source:
    Many methods based on adg.stanford.edu, see methods for details
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
        self.tag = 'Supersonic_OpenVSP_Wave_Drag'
        
        # correction factors
        settings =  self.settings
        settings.fuselage_lift_correction           = 1.14
        settings.trim_drag_correction_factor        = 1.02
        settings.wing_parasite_drag_form_factor     = 1.1
        settings.fuselage_parasite_drag_form_factor = 2.3
        settings.aircraft_span_efficiency_factor    = 0.78
        settings.span_efficiency                    = None
        settings.viscous_lift_dependent_drag_factor = 0.38
        settings.drag_coefficient_increment         = 0.0000
        settings.oswald_efficiency_factor           = None
        settings.maximum_lift_coefficient           = np.inf
        settings.number_slices                      = 20
        settings.number_rotations                   = 10
        
        # vortex lattice configurations
        settings.number_spanwise_vortices = 5
        settings.number_chordwise_vortices = 1
        
        
        # build the evaluation process
        compute = self.process.compute
        
        compute.lift = Process()
        compute.lift.inviscid_wings                = Vortex_Lattice()
        compute.lift.fuselage                      = Common.Lift.fuselage_correction
        compute.lift.total                         = Common.Lift.aircraft_total
        
        compute.drag = Process()
        compute.drag.compressibility               = Process()
        compute.drag.compressibility.total         = VSP_Methods.compressibility_drag_total       
        compute.drag.parasite                      = Process()
        compute.drag.parasite.wings                = Process_Geometry('wings')
        compute.drag.parasite.wings.wing           = Common.Drag.parasite_drag_wing 
        compute.drag.parasite.fuselages            = Process_Geometry('fuselages')
        compute.drag.parasite.fuselages.fuselage   = Common.Drag.parasite_drag_fuselage
        compute.drag.parasite.nacelles             = Process_Geometry('nacelles')
        compute.drag.parasite.nacelles.nacelle     = Methods.Drag.parasite_drag_nacelle
        #compute.drag.parasite.pylons               = Methods.Drag.parasite_drag_pylon # supersonic pylon methods not currently available
        compute.drag.parasite.total                = Common.Drag.parasite_total
        compute.drag.induced                       = Methods.Drag.induced_drag_aircraft
        compute.drag.miscellaneous                 = Methods.Drag.miscellaneous_drag_aircraft
        compute.drag.untrimmed                     = Common.Drag.untrimmed
        compute.drag.trim                          = Common.Drag.trim
        compute.drag.total                         = Methods.Drag.total_aircraft
        
        
    def initialize(self):
        """Initializes the surrogate needed for lift calculation and removes old volume drag data files.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        self.geometry.tag (geometry in full is also attached to a process)
        """    
        super(Supersonic_OpenVSP_Wave_Drag, self).initialize()
        import os
        
        # Remove old volume drag data so that new data can be appended without issues
        try:
            os.remove('volume_drag_data_' + self.geometry.tag + '.npy')  
        except:
            pass
        
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

        self.process.compute.lift.inviscid_wings.geometry = self.geometry
        self.process.compute.lift.inviscid_wings.initialize(use_surrogate,n_sw,n_cw,propeller_wake_model,use_bemt_wake_model,ito,wdt,nwts,mf,mn)
        
    finalize = initialize        