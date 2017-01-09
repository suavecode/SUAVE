# SU2_Euler_Super.py
#
# Created:  Dec 2016, T. MacDonald
# Modified: Jan 2017, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data
from Markup import Markup
from SUAVE.Analyses import Process
import numpy as np

from SUAVE.Input_Output.OpenVSP.write_vsp_mesh import write_vsp_mesh
from SUAVE.Input_Output.GMSH.write_geo_file import write_geo_file
from SUAVE.Input_Output.GMSH.mesh_geo_file import mesh_geo_file

# Default aero Results
from Results import Results

# The aero methods
from SUAVE.Methods.Aerodynamics import Supersonic_Zero as Methods
from SUAVE.Methods.Aerodynamics import Fidelity_Zero   as FZ_Methods
from Process_Geometry import Process_Geometry
from SUAVE.Analyses.Aerodynamics.SU2_inviscid_Super import SU2_inviscid_Super

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------
class SU2_Euler_Super(Markup):
    
    def __defaults__(self):
        
        self.tag    = 'SU2_Euler_Super_markup'       
    
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
        settings.half_mesh_flag                     = True
        settings.parallel                           = False
        settings.processors                         = 1
        settings.vsp_mesh_growth_ratio              = 1.3
        
        # Build the evaluation process
        compute = self.process.compute
        compute.lift = Process()

        # Run SU2
        compute.lift.inviscid                         = SU2_inviscid_Super()
        compute.lift.total                            = SUAVE.Methods.Aerodynamics.AERODAS.AERODAS_setup.lift_total
        
        # Do a traditional drag buildup
        compute.drag = Process()
        compute.drag.compressibility               = Process()
        compute.drag.compressibility.total         = Methods.Drag.compressibility_drag_total      
        compute.drag.parasite                      = Process()
        compute.drag.parasite.wings                = Process_Geometry('wings')
        compute.drag.parasite.wings.wing           = Methods.Drag.parasite_drag_wing
        compute.drag.parasite.fuselages            = Process_Geometry('fuselages')
        compute.drag.parasite.fuselages.fuselage   = Methods.Drag.parasite_drag_fuselage
        compute.drag.parasite.propulsors           = Process_Geometry('propulsors')
        compute.drag.parasite.propulsors.propulsor = Methods.Drag.parasite_drag_propulsor
        #compute.drag.parasite.pylons               = Methods.Drag.parasite_drag_pylon # currently unavailable for supersonic
        compute.drag.parasite.total                = Methods.Drag.parasite_total
        compute.drag.induced                       = Methods.Drag.induced_drag_aircraft
        compute.drag.miscellaneous                 = Methods.Drag.miscellaneous_drag_aircraft
        compute.drag.untrimmed                     = SUAVE.Methods.Aerodynamics.SU2_Euler.untrimmed
        compute.drag.trim                          = Methods.Drag.trim
        compute.drag.spoiler                       = FZ_Methods.Drag.spoiler_drag
        compute.drag.total                         = SUAVE.Methods.Aerodynamics.SU2_Euler.total_aircraft_drag
        
        
    def initialize(self):
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