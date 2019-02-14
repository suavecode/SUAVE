## @ingroup Analyses-Aerodynamics
# Blown_Wing_Vortex_Lattice.py
#
# Created:  Jan 2018, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
import SUAVE

from SUAVE.Core import Data
from SUAVE.Core import Units
from SUAVE.Methods.Aerodynamics.Blown_Wing_Aero  import blown_wing_weissinger_vortex_lattice

# local imports
from .Aerodynamics import Aerodynamics

# package imports
import numpy as np


# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------
## @ingroup Analyses-Aerodynamics
class Blown_Wing_Vortex_Lattice(Aerodynamics):
    """This model incorperate the propeller wake of propellers in the estimation of wing lift vortex lattice.
    Unlike Vortex_Lattice.py which uses a surrogate, a direct computation of aerodynamic properties at each 
    colocation point in each segment is used. 

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
        self.tag = 'Blown_Wing_Vortex_Lattice'

        self.geometry = Data()
        self.settings = Data()

        # correction factors
        self.settings.fuselage_lift_correction           = 1.14
        self.settings.trim_drag_correction_factor        = 1.02
        self.settings.wing_parasite_drag_form_factor     = 1.1
        self.settings.fuselage_parasite_drag_form_factor = 2.3
        self.settings.aircraft_span_efficiency_factor    = 0.78
        self.settings.drag_coefficient_increment         = 0.0000        
        self.index = 0
    
    def evaluate(self,state,settings,geometry):
        # unpack
        conditions             = state.conditions
        propulsors             = geometry.propulsors
        vehicle_reference_area = geometry.reference_area
        total_lift_coeff       = Data()                
        n                                                          = state.numerics.number_control_points        
        total_lift_coeff                                           = np.zeros((n,1))
        inviscid_wings_lift                                        = Data()           
        conditions.aerodynamics.lift_breakdown.inviscid_wings_lift = Data()
        state.conditions.aerodynamics.lift_coefficient_wing        = Data()
        
        for wing in geometry.wings.keys():
            wing_CL = np.zeros((n,1))
            for index in range(n):
                [wing_lift_coeff,  wing_drag_coeff ]    = blown_wing_weissinger_vortex_lattice(conditions,settings,geometry.wings[wing],propulsors,0)
                wing_CL[index]                          = wing_lift_coeff      
                
            inviscid_wings_lift[wing]                                        = wing_CL            
            conditions.aerodynamics.lift_breakdown.inviscid_wings_lift[wing] = inviscid_wings_lift[wing]   
            state.conditions.aerodynamics.lift_coefficient_wing[wing]        = inviscid_wings_lift[wing] 
            total_lift_coeff                                                 += wing_CL * geometry.wings[wing].areas.reference / vehicle_reference_area                         
        
        conditions.aerodynamics.lift_breakdown.inviscid_wings_lift.total = total_lift_coeff
        state.conditions.aerodynamics.lift_coefficient                   = total_lift_coeff
        state.conditions.aerodynamics.inviscid_lift                      = total_lift_coeff 
        inviscid_wings_lift.total                                        = total_lift_coeff   
        
        return inviscid_wings_lift
    

