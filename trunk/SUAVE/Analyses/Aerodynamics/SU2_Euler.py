## @ingroup Analyses-Aerodynamics
# SU2_Euler.py
#
# Created:  Sep 2016, E. Botero
# Modified: Jan 2017, T. MacDonald
#           Apr 2019, T. MacDonald
#           Mar 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data
from .Markup import Markup
from SUAVE.Analyses import Process
import numpy as np

from SUAVE.Input_Output.OpenVSP.write_vsp_mesh import write_vsp_mesh
from SUAVE.Input_Output.GMSH.write_geo_file import write_geo_file
from SUAVE.Input_Output.GMSH.mesh_geo_file import mesh_geo_file

# The aero methods
from SUAVE.Methods.Aerodynamics.Common import Fidelity_Zero as Common
from .Process_Geometry import Process_Geometry
from SUAVE.Analyses.Aerodynamics.SU2_inviscid import SU2_inviscid

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------
## @ingroup Analyses-Aerodynamics
class SU2_Euler(Markup):
    """This uses SU2 to compute lift.

    Assumptions:
    Subsonic

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
        self.tag    = 'SU2_Euler_markup'       
    
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
        settings.maximum_lift_coefficient           = np.inf 
        settings.half_mesh_flag                     = True
        settings.parallel                           = False
        settings.processors                         = 1
        settings.vsp_mesh_growth_ratio              = 1.3
        settings.vsp_mesh_growth_limiting_flag      = False
        settings.recalculate_total_wetted_area      = False
        
        
        # Build the evaluation process
        compute = self.process.compute
        compute.lift = Process()

        # Run SU2 to determine lift
        compute.lift.inviscid                      = SU2_inviscid()
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
        
        
    def initialize(self):
        """Initializes the surrogate needed for SU2, including building the surface and volume meshes.

        Assumptions:
        Vehicle is available in OpenVSP
        
        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        self.geometry.tag               <string> (geometry is also set as part of the lift process)
        self.process.compute.lift.
          inviscid.training_file        (optional - determines if new SU2 runs are necessary)
        self.settings.
          half_mesh_flag                <boolean> Determines if a symmetry plane is used
          vsp_mesh_growth_ratio         [-] Determines how the mesh grows
          vsp_mesh_growth_limiting_flag <boolean> Determines if 3D growth limiting is used
        """         
        super(SU2_Euler, self).initialize()
        self.process.compute.lift.inviscid.geometry = self.geometry
        
        tag = self.geometry.tag
        # Mesh the geometry in prepartion for CFD if no training file exists
        if self.process.compute.lift.inviscid.training_file is None:
            write_vsp_mesh(self.geometry,tag,self.settings.half_mesh_flag,self.settings.vsp_mesh_growth_ratio,self.settings.vsp_mesh_growth_limiting_flag)
            write_geo_file(tag)
            mesh_geo_file(tag)
        
        # Generate the surrogate
        self.process.compute.lift.inviscid.initialize()
        
    finalize = initialize