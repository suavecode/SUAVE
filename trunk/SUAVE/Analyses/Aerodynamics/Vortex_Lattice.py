## @ingroup Analyses-Aerodynamics
# Vortex_Lattice.py
#
# Created:  Nov 2013, T. Lukaczyk
# Modified:     2014, T. Lukaczyk, A. Variyar, T. Orra
#           Feb 2016, A. Wendorff
#           Apr 2017, T. MacDonald
#           Nov 2017, E. Botero


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
import SUAVE

from SUAVE.Core import Data
from SUAVE.Core import Units

from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift import weissinger_vortex_lattice

# local imports
from Aerodynamics import Aerodynamics

# package imports
import numpy as np


# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------
## @ingroup Analyses-Aerodynamics
class Vortex_Lattice(Aerodynamics):
    """This builds a surrogate and computes lift using a basic vortex lattice.

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
        self.tag = 'Vortex_Lattice'

        self.geometry = Data()
        self.settings = Data()

        # correction factors
        self.settings.fuselage_lift_correction           = 1.14
        self.settings.trim_drag_correction_factor        = 1.02
        self.settings.wing_parasite_drag_form_factor     = 1.1
        self.settings.fuselage_parasite_drag_form_factor = 2.3
        self.settings.aircraft_span_efficiency_factor    = 0.78
        self.settings.drag_coefficient_increment         = 0.0000

    def evaluate(self,state,settings,geometry):
        # unpack
        conditions = state.conditions
        #propulsion = geometry.propulsors['network'].propeller
        propulsion = geometry.propulsors['turbofan'] 
        
        # store model for lift coefficients of each wing
        state.conditions.aerodynamics.lift_coefficient_wing             = Data()   
        total_lift_coeff = 0
        total_drag_coeff = 0
        total_lift       = 0
        total_drag       = 0 
        
        for wing in geometry.wings.values():
            # run vortex lattice at quaried flight conditions
            [wing_lift,wing_lift_coeff,wing_drag,wing_drag_coeff] = weissinger_vortex_lattice(conditions,settings,wing,propulsion)
            conditions.aerodynamics.lift_breakdown.inviscid_wings_lift[wing] = inviscid_wings_lift[wing]
            state.conditions.aerodynamics.lift_coefficient_wing[wing]        = inviscid_wings_lift[wing]
               
            # lift 
            total_lift_coeff += wing_lift_coeff * wing.areas.reference / vehicle_reference_area
            total_lift  += wing_lift  
            
            # inviscid drag 
            total_drag_coeff += wing_drag_coeff * wing.areas.reference / vehicle_reference_area
            total_drag  += wing_drag  
            
        # inviscid lift of wings only
        conditions.aerodynamics.lift_breakdown.inviscid_wings_lift       = Data()
        conditions.aerodynamics.lift_breakdown.inviscid_wings_lift.total = total_lift_coeff
        state.conditions.aerodynamics.lift_coefficient                   = total_lift_coeff

