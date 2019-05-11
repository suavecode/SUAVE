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
from .Aerodynamics import Aerodynamics

# package imports
import numpy as np


# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------
## @ingroup Analyses-Aerodynamics
class Vortex_Lattice_No_Surrogate(Aerodynamics):
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

        # vortex lattice configurations
        self.settings.number_panels_spanwise = 10
        
    def initialize(self):
        """Drives functions to get training samples and build a surrogate.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """                      
        # sample training data
        # Build the vortex distribution
        
        pass


    def evaluate(self,state,settings,geometry):
        """Evaluates lift and drag using available surrogates.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        state.conditions.
          freestream.dynamics_pressure       [-]
          angle_of_attack                    [radians]

        Outputs:
        conditions.aerodynamics.lift_breakdown.
          inviscid_wings_lift[wings.*.tag]   [-] CL (wing specific)
          inviscid_wings_lift.total          [-] CL
        conditions.aerodynamics.
          lift_coefficient_wing              [-] CL (wing specific)
        inviscid_wings_lift                  [-] CL

        Properties Used:
        self.surrogates.
          lift_coefficient                   [-] CL
          wing_lift_coefficient[wings.*.tag] [-] CL (wing specific)
        """          
        """ process vehicle to setup geometry, condititon and settings
            Inputs:
                conditions - DataDict() of aerodynamic conditions
            Outputs:
                CL - array of lift coefficients, same size as alpha
                CD - array of drag coefficients, same size as alpha
            Assumptions:
                linear intperolation surrogate model on Mach, Angle of Attack
                    and Reynolds number
                locations outside the surrogate's table are held to nearest data
                no changes to initial geometry or settings
        """
        
        # unpack        
        conditions =  state.conditions
        
        # inviscid lift of wings only
        inviscid_wings_lift                                              = Data()
        inviscid_wings_lift.total, wing_lifts                            = calculate_lift_vortex_lattice(conditions,settings,geometry)
        conditions.aerodynamics.lift_breakdown.inviscid_wings_lift       = Data()
        conditions.aerodynamics.lift_breakdown.inviscid_wings_lift.total = inviscid_wings_lift.total
        state.conditions.aerodynamics.lift_coefficient                   = inviscid_wings_lift.total
        state.conditions.aerodynamics.lift_coefficient_wing              = wing_lifts
        
        return inviscid_wings_lift

# ----------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------


def calculate_lift_vortex_lattice(conditions,settings,geometry):
    """Calculate the total vehicle lift coefficient and specific wing coefficients (with specific wing reference areas)
    using a vortex lattice method.

    Assumptions:
    None

    Source:
    N/A

    Inputs:
    conditions                      (passed to vortex lattice method)
    settings                        (passed to vortex lattice method)
    geometry.reference_area         [m^2]
    geometry.wings.*.reference_area (each wing is also passed to the vortex lattice method)

    Outputs:
    

    Properties Used:
    
    """            

    # unpack
    vehicle_reference_area = geometry.reference_area

    # iterate over wings
    total_lift_coeff = 0.0
    wing_lifts = Data()

    for wing in geometry.wings.values():

        [wing_lift_coeff,wing_drag_coeff] = weissinger_vortex_lattice(conditions,settings,wing)
        total_lift_coeff += wing_lift_coeff * wing.areas.reference / vehicle_reference_area
        wing_lifts[wing.tag] = wing_lift_coeff

    return total_lift_coeff, wing_lifts
