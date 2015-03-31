# Fidelity_Zero.py
#
# Created:  Trent, Nov 2013
# Modified: Trent, Anil, Tarik, Feb 2014
# Modified: Trent, Jan 2014


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
import SUAVE

from SUAVE.Core import Data
from SUAVE.Core import Units

from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift import weissinger_vortex_lattice
from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift import compute_aircraft_lift
from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Drag import compute_aircraft_drag

# local imports
from Aerodynamics import Aerodynamics
from Results      import Results

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np
import scipy as sp


# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Inviscid_Wings_Lift(Aerodynamics):
    """ SUAVE.Analyses.Aerodynamics.Fidelity_Zero
        aerodynamic model that builds a surrogate model for clean wing
        lift, using vortex lattice, and various handbook methods
        for everything else

        this class is callable, see self.__call__

    """

    def __defaults__(self):

        self.tag = 'Fidelity_Zero'

        self.geometry = Data()
        self.settings = Data()



    def evaluate(self,state,settings,geometry):
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
        settings   = self.settings
        #geometry   = self.geometry

        surrogates = self.surrogates

        conditions = state.conditions
        
        q    = conditions.freestream.dynamic_pressure
        AoA  = conditions.aerodynamics.angle_of_attack
        Sref = geometry.reference_area
        
        
        
        # inviscid lift of wings only
        inviscid_wings_lift = 2*np.pi*AoA 
        conditions.aerodynamics.lift_breakdown.inviscid_wings_lift = inviscid_wings_lift
        state.conditions.aerodynamics.lift_coefficient = inviscid_wings_lift

        return inviscid_wings_lift


