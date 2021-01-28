## @ingroup Analyses-Weights
# Weights_Transport.py
#
# Created:  Apr 2017, Matthew Clarke
# Modified: Oct 2017, T. MacDonald
#           Apr 2020, E. Botero

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
class Weights_Transport(Weights):
    """ This is class that evaluates the weight of Transport class aircraft

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

    def __defaults__(self):
        """This sets the default values and methods for the tube and wing
        aircraft weight analysis.

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
        self.tag = 'transport'

        self.vehicle  = Data()
        self.settings = Data()
        self.settings.weight_reduction_factors = Data()

        # Reduction factors are proportional (.1 is a 10% weight reduction)
        self.settings.weight_reduction_factors.main_wing = 0.
        self.settings.weight_reduction_factors.fuselage  = 0.
        self.settings.weight_reduction_factors.empennage = 0.  # applied to horizontal and vertical stabilizers
        
        # FLOPS settings
        self.settings.FLOPS = Data()
        # Aeroelastic tailoring factor [0 no aeroelastic tailoring, 1 maximum aeroelastic tailoring]
        self.settings.FLOPS.aeroelastic_tailoring_factor = 0.
        # Wing strut bracing factor [0 for no struts, 1 for struts]
        self.settings.FLOPS.strut_braced_wing_factor     = 0.
        # Composite utilization factor [0 no composite, 1 full composite]
        self.settings.FLOPS.composite_utilization_factor = 0.5
        
        # Raymer settings
        self.settings.Raymer = Data()
        self.settings.Raymer.fuselage_mounted_landing_gear_factor = 1. # 1. if false, 1.12 if true

    def evaluate(self, method="New SUAVE"):
        """Evaluate the weight analysis.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        results

        Properties Used:
        N/A
        """
        # unpack
        vehicle = self.vehicle
        results = SUAVE.Methods.Weights.Correlations.Common.empty_weight(vehicle, settings=self.settings,
                                                                         method_type=method)

        # storing weigth breakdown into vehicle
        vehicle.weight_breakdown = results

        # updating empty weight
        vehicle.mass_properties.operating_empty = results.empty

        # done!
        return results