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
from SUAVE.Methods.Aerodynamics.Blown_Wing_Aero  import blown_wing_VLM

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
        aero                   = state.conditions.aerodynamics
        propulsors             = geometry.propulsors
        vehicle_reference_area = geometry.reference_area        
        n                      = state.numerics.number_control_points   
        
        # lift
        total_lift_coeff                                     = np.zeros((n,1))
        inviscid_wings_lift                                  = Data()
        inviscid_wings_lift_distribution                     = Data()
        aero.lift_breakdown.inviscid_wings_lift              = Data()
        aero.lift_breakdown.inviscid_wings_lift_distribution = Data()        
        aero.lift_coefficient_wing                           = Data()
         
        # drag  
        total_inviscid_drag_coeff                             = np.zeros((n,1))
        inviscid_wings_drag                                   = Data() 
        inviscid_wings_drag_distribution                      = Data()
        aero.drag_breakdown.inviscid_wings_drag               = Data()
        aero.drag_breakdown.inviscid_wings_drag_distribution  = Data() 
        
        for wing in geometry.wings.keys():
            wing_CL       = np.zeros((n,1))
            wing_Cl_dist  = np.zeros((n,settings.number_panels_spanwise))
            wing_CDi      = np.zeros((n,1))
            wing_Cdi_dist = np.zeros((n,settings.number_panels_spanwise))            
            for index in range(n):
                [wing_lift_coeff,  wing_inviscid_drag_coeff , wing_lift_distribution , wing_inviscid_drag_distribution ] = blown_wing_VLM(conditions,settings,geometry.wings[wing],propulsors, index )
                wing_CL[index]       = wing_lift_coeff      
                wing_Cl_dist[index]  = wing_lift_distribution  
                wing_CDi[index]      = wing_inviscid_drag_coeff     
                wing_Cdi_dist[index] = wing_inviscid_drag_distribution  
             
            # lift   
            inviscid_wings_lift[wing]                                  = wing_CL   
            inviscid_wings_lift_distribution[wing]                     = wing_Cl_dist
            aero.lift_breakdown.inviscid_wings_lift[wing]              = inviscid_wings_lift[wing]   
            aero.lift_breakdown.inviscid_wings_lift_distribution[wing] = inviscid_wings_lift_distribution[wing] 
            aero.lift_coefficient_wing[wing]                           = inviscid_wings_lift[wing] 
            total_lift_coeff                                           += wing_CL * geometry.wings[wing].areas.reference / vehicle_reference_area                         
       
            # drag  
            inviscid_wings_drag[wing]                                  = wing_CDi   
            inviscid_wings_drag_distribution[wing]                     = wing_Cdi_dist
            aero.drag_breakdown.inviscid_wings_drag[wing]              = inviscid_wings_drag[wing]   
            aero.drag_breakdown.inviscid_wings_drag_distribution[wing] = inviscid_wings_drag_distribution[wing] 
            aero.lift_coefficient_wing[wing]                           = inviscid_wings_drag[wing]            
            total_inviscid_drag_coeff                                  += wing_CDi * geometry.wings[wing].areas.reference / vehicle_reference_area                              
        
        # pack 
        # lift 
        aero.lift_breakdown.inviscid_wings_lift.total = total_lift_coeff
        aero.lift_coefficient                         = total_lift_coeff
        aero.inviscid_lift                            = total_lift_coeff 
        inviscid_wings_lift.total                     = total_lift_coeff   
        
        # drag
        aero.drag_breakdown.inviscid_wings_drag.total = total_inviscid_drag_coeff  
        
        return inviscid_wings_lift
    

