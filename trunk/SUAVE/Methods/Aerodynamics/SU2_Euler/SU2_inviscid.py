# SU2_inviscid.py
#
# Created:  Sep 2016, E. Botero
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
import SUAVE

from SUAVE.Core import Data
from SUAVE.Core import Units

# local imports
from Aerodynamics import Aerodynamics

# package imports
import numpy as np
import scipy as sp

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class SU2_inviscid(Aerodynamics):
    """ SUAVE.Analyses.Aerodynamics.Fidelity_Zero
        aerodynamic model that builds a surrogate model for a model,
        using SU2 Euler, and various handbook methods
        for everything else

        this class is callable, see self.__call__

    """

    def __defaults__(self):

        self.tag = 'SU2_inviscid'

        self.geometry = Data()
        self.settings = Data()

        # conditions table, used for surrogate model training
        self.training = Data()        
        self.training.angle_of_attack  = np.array([-10.,-5.,0.,5.,10.]) * Units.deg
        self.training.Mach             = np.array([0.05,0.3,0.5,0.8])
        self.training.lift_coefficient = None
        self.training.drag_coefficient = None
        
        # surrogate model
        self.surrogates = Data()
        self.surrogates.coefficients   = None
 
        
    def initialize(self):
                   
        # sample training data
        self.sample_training()
                    
        # build surrogate
        self.build_surrogate()


    def evaluate(self,state,settings,geometry):
        """ process vehicle to setup geometry, condititon and settings

            Inputs:
                conditions - DataDict() of aerodynamic conditions

            Outputs:
                CL - array of lift coefficients, same size as alpha
                CD - array of drag coefficients, same size as alpha

            Assumptions:
                non-linear intperolation surrogate model on Mach, Angle of Attack
                    and Reynolds number
                locations outside the surrogate's table are held to nearest data
                no changes to initial geometry or settings
        """

        # unpack
        surrogates = self.surrogates        
        conditions = state.conditions
        
        # unpack
        mach = conditions.freestream.mach_number
        AoA  = conditions.aerodynamics.angle_of_attack
        lift_model = surrogates.lift_coefficient
        drag_model = surrogates.drag_coefficient
        
        # inviscid lift
        inviscid_lift                                              = lift_model(AoA,mach)
        conditions.aerodynamics.lift_breakdown.inviscid_wings_lift = inviscid_lift
        state.conditions.aerodynamics.lift_coefficient             = inviscid_lift
        
        # inviscid drag
        inviscid_drag                                              = drag_model(AoA,mach)
        state.conditions.aerodynamics.inviscid_drag_coefficient    = inviscid_drag

        return inviscid_lift, inviscid_drag


    def sample_training(self):
        
        # unpack
        geometry = self.geometry
        settings = self.settings
        training = self.training
        
        AoA  = training.angle_of_attack
        mach = training.Mach 
        CL   = np.zeros(len(AoA),len(mach))
        CD   = np.zeros(len(AoA),len(mach))

        # condition input, local, do not keep
        konditions              = Data()
        konditions.aerodynamics = Data()

        # calculate aerodynamics for table
        
        for i,_ in enumerate(AoA):
            for j in enumerate(mach):
                
                # overriding conditions, thus the name mangling
                konditions.aerodynamics.angle_of_attack = AoA[i]
                konditions.aerodynamics.mach            = mach[j]
                
                # these functions are inherited from Aerodynamics() or overridden
                CL[i,j],CD[i,j] = call_SU2(konditions, settings, geometry)

        # store training data
        training.coefficients = [CL,CD]

        return

    def build_surrogate(self):

        # unpack data
        training  = self.training
        AoA_data  = training.angle_of_attack
        mach_data = training.Mach
        CL_data   = training.coefficients[0,:]
        CD_data   = training.coefficients[1,:]
        
        # Some combo of mach and AoA
        #xy = 
        
        # learn the model
        cl_surrogate = sp.interpolate.CloughTocher2DInterpolator(xy,CL_data)
        cd_surrogate = sp.interpolate.CloughTocher2DInterpolator(xy,CL_data)

        self.surrogates.lift_coefficient = cl_surrogate
        self.surrogates.drag_coefficient = cd_surrogate

        return



# ----------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------

def call_SU2(conditions,settings,geometry):
    """ calculate total vehicle lift coefficient by SU2
    """

    lift_coefficient = 1.
    drag_coefficient = 1.

    return lift_coefficient, drag_coefficient
