# Linear_Lift.py
#
# Created:  Trent, Nov 2013
# Modified: Trent, Anil, Tarik, Feb 2014
# Modified: Trent, Jan 2014 
# Modified: Feb 2016, Andrew Wendorff


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
import SUAVE

from SUAVE.Core import Data


# local imports
from Aerodynamics import Aerodynamics

# package imports
import numpy as np
from numpy import pi

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Linear_Lift(Aerodynamics):
    """ SUAVE.Analyses.Aerodynamics.Linear_Lift
        aerodynamic model that builds a surrogate model for clean wing
        lift, using vortex lattice, and various handbook methods
        for everything else

        this class is callable, see self.evaluate

    """

    def __defaults__(self):

        self.tag = 'Fidelity_Zero'

        self.geometry = Data()
        self.settings = Data()
        
        self.settings.zero_lift_coefficient = 0.0
        self.settings.slope_correction_coefficient = 1.0
        
        #self.settings.span = 1.0
        #self.settings.sweep_angle  = 0.0
        



    def evaluate(self,state,settings=None,geometry=None):
        """ process vehicle to setup geometry, condititon and settings
        
            Settings:
                zero_lift_coefficient = 0.0
                slope_correction_coefficient = 1.0

            Inputs:
                state - a data dictionary with fields
                   conditions.aerodynamics.angle_of_attack

            Outputs:
                CL - inviscid lift coefficient 

            Assumptions:
                that standard 2 pi alpha thing, anil fix this
        """

        # unpack
        settings   = self.settings
        conditions = state.conditions
        alpha      = conditions.aerodynamics.angle_of_attack
        e          = settings.slope_correction_coefficient
        CL0        = settings.zero_lift_coefficient
        
        # inviscid lift of wings only
        CL = 2.0 * pi * alpha * e + CL0
        
        # pack
        inviscid_wings_lift = CL
        conditions.aerodynamics.lift_breakdown.inviscid_wings_lift.total = inviscid_wings_lift
        conditions.aerodynamics.lift_coefficient                         = inviscid_wings_lift

        return inviscid_wings_lift


