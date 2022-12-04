## @ingroup Analyses-Weights

# Weights_eVTOL.py
#
# Created:  Aug, 2017, J. Smart
# Modified: Apr, 2018, J. Smart
#           May, 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data
from .Weights import Weights


# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

## @ingroup Analyses-Weights
class Weights_eVTOL(Weights):
    """This is class that evaluates the weight of an eVTOL aircraft
    
    Assumptions:
    None
    
    Source:
    N/A
    
    Inputs:
    N/A
    
    Outputs:
    N/A
    
    Properties Used:
    N/A
    """

    def __defaults__(self):
        """Sets the default parameters for an eVTOL weight analysis

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        N/A

        Outputs:
        N/A

        Properties Used:
        N/A
        """
        
        self.tag = 'weights_evtol'
        
        self.vehicle  = Data()
        self.settings = Data()
        
        self.settings.empty = SUAVE.Methods.Weights.Buildups.eVTOL.empty


        # Methodological Settings

        self.settings.method_settings = Data()

        self.settings.method_settings.contingency_factor            = 1.1,
        self.settings.method_settings.speed_of_sound                = 340.294,
        self.settings.method_settings.max_tip_mach                  = 0.65,
        self.settings.method_settings.disk_area_factor              = 1.15,
        self.settings.method_settings.safety_factor                 = 1.5,
        self.settings.method_settings.max_thrust_to_weight_ratio    = 1.1,
        self.settings.method_settings.max_g_load                    = 3.8,
        self.settings.method_settings.motor_efficiency              = 0.85*0.98

    def evaluate(self,conditions=None):
        """Evaluates the weight of an eVTOL aircraft

        Assumptions:
        None

        Inputs:
        self.settings.method_settings.      [Data]
            contingency_factor              [Float] Secondary Weight Est.
            speed_of_sound                  [Float] Design Point Speed of Sound
            max_tip_mach                    [Float] Max Rotor Tip Mach Number
            disk_area_factor                [Float] Disk Area Factor (ref. Johnson 2-6.2)
            safety_factor                   [Float] Structural Factor of Safety
            max_thrust_to_weight_ratio      [Float] Design T/W Ratio
            max_g_load                      [Float] Design G Load
            motor_efficiency                [Float] Design Point Motor Efficiency

        Outputs:
        N/A

        Properties Used:
        N/A
        """

        vehicle = self.vehicle
        results = self.settings.empty(vehicle,
                                      settings=self.settings)

        vehicle.weight_breakdown = results

        vehicle.mass_properties.operating_empty = results.empty

        return results