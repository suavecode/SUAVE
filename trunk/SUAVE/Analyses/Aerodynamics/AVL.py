## @ingroup Analyses-Aerodynamics
# AVL.py
#
# Created:  Apr 2017, M. Clarke 
# Modified: Apr 2019, T. MacDonald 
#           Mar 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data
from .Markup import Markup
from SUAVE.Analyses import Process
import numpy as np

# The aero methods
from SUAVE.Methods.Aerodynamics.Common import Fidelity_Zero as Common
from .Process_Geometry import Process_Geometry
from SUAVE.Analyses.Aerodynamics.AVL_Inviscid import AVL_Inviscid

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------
## @ingroup Analyses-Aerodynamics
class AVL(Markup):
    """This uses AVL to compute lift.

    Assumptions:
    None

    Source:
    None
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
        self.tag    = 'AVL_markup'       
    
        # Correction factors
        settings = self.settings
        settings.trim_drag_correction_factor        = 1.02
        settings.wing_parasite_drag_form_factor     = 1.1
        settings.fuselage_parasite_drag_form_factor = 2.3
        settings.oswald_efficiency_factor           = None
        settings.span_efficiency                    = None
        settings.viscous_lift_dependent_drag_factor = 0.38
        settings.drag_coefficient_increment         = 0.0000
        settings.spoiler_drag_increment             = 0.00 
        settings.recalculate_total_wetted_area      = False
        
        # ------
        settings.number_spanwise_vortices           = 20
        settings.number_chordwise_vortices          = 10    
        settings.keep_files                         = False
        settings.save_regression_results            = False          
        settings.regression_flag                    = False   
        settings.trim_aircraft                      = False 
        settings.print_output                       = False   
        
        settings.maximum_lift_coefficient           = np.inf  
        settings.side_slip_angle                    = 0.0
        settings.roll_rate_coefficient              = 0.0
        settings.pitch_rate_coefficient             = 0.0
        settings.lift_coefficient                   = None
                
        # Build the evaluation process
        compute = self.process.compute
        compute.lift = Process()

        # Run AVL to determine lift
        compute.lift.inviscid                      = AVL_Inviscid()
        compute.lift.total                         = Common.Lift.aircraft_total
        
        # Do a traditional drag buildup
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
        compute.drag.compressibility               = Process()
        compute.drag.compressibility.wings         = Process_Geometry('wings')
        compute.drag.compressibility.wings.wing    = Common.Drag.compressibility_drag_wing
        compute.drag.compressibility.total         = Common.Drag.compressibility_drag_wing_total        
        compute.drag.miscellaneous                 = Common.Drag.miscellaneous_drag_aircraft_ESDU
        compute.drag.untrimmed                     = Common.Drag.untrimmed
        compute.drag.trim                          = Common.Drag.trim
        compute.drag.spoiler                       = Common.Drag.spoiler_drag
        compute.drag.total                         = Common.Drag.total_aircraft
        
        
    def initialize(self):
        """Initializes the surrogate needed for AVL.

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
        super(AVL, self).initialize()
        # unpack
        sv  = self.settings.number_spanwise_vortices
        cv  = self.settings.number_chordwise_vortices 
        kf  = self.settings.keep_files
        srr = self.settings.save_regression_results
        rf  = self.settings.regression_flag
        po  = self.settings.print_output 
        ta  = self.settings.trim_aircraft
        ssa = self.settings.side_slip_angle
        rrc = self.settings.roll_rate_coefficient
        pra = self.settings.pitch_rate_coefficient
        lc  = self.settings.lift_coefficient              
        
        self.process.compute.lift.inviscid.geometry = self.geometry
        
        # Generate the surrogate
        self.process.compute.lift.inviscid.initialize(sv,cv,kf,srr,rf,po,ta,ssa,rrc,pra,lc)
        
    finalize = initialize
    
    
