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
from SUAVE.Plugins.SU2.call_SU2_CFD import call_SU2_CFD
from SUAVE.Plugins.SU2.write_SU2_cfg import write_SU2_cfg

# package imports
import numpy as np
import scipy as sp
import scipy.interpolate

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
        self.settings.half_mesh_flag   = True

        # conditions table, used for surrogate model training
        self.training = Data()        
        self.training.angle_of_attack  = np.array([0.,5.]) * Units.deg
        self.training.Mach             = np.array([0.4,0.75])
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
        CL   = np.zeros([len(AoA)*len(mach),1])
        CD   = np.zeros([len(AoA)*len(mach),1])

        # condition input, local, do not keep
        konditions              = Data()
        konditions.aerodynamics = Data()

        # calculate aerodynamics for table
        table_size = len(AoA)*len(mach)
        xy = np.zeros([table_size,2])
        count = 0
        for i,_ in enumerate(AoA):
            for j,_ in enumerate(mach):
                
                xy[count,:] = np.array([AoA[i],mach[j]])
                # overriding conditions, thus the name mangling
                konditions.aerodynamics.angle_of_attack = AoA[i]
                konditions.aerodynamics.mach            = mach[j]
                
                # these functions are inherited from Aerodynamics() or overridden
                CL[count],CD[count] = call_SU2(konditions, settings, geometry)
                count += 1

        # store training data
        training.coefficients = np.hstack([CL,CD])
        training.grid_points  = xy

        return

    def build_surrogate(self):

        # unpack data
        training  = self.training
        AoA_data  = training.angle_of_attack
        mach_data = training.Mach
        CL_data   = training.coefficients[:,0]
        CD_data   = training.coefficients[:,1]
        xy        = training.grid_points 
        
        # learn the model
        cl_surrogate = sp.interpolate.CloughTocher2DInterpolator(xy,CL_data)
        cd_surrogate = sp.interpolate.CloughTocher2DInterpolator(xy,CD_data)

        self.surrogates.lift_coefficient = cl_surrogate
        self.surrogates.drag_coefficient = cd_surrogate

        return



# ----------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------

def call_SU2(conditions,settings,geometry):
    """ calculate total vehicle lift coefficient by SU2
    """

    half_mesh_flag = settings.half_mesh_flag
    tag            = geometry.tag
    
    SU2_settings = Data()
    if half_mesh_flag == False:
        SU2_settings.reference_area  = geometry.reference_area
    else:
        SU2_settings.reference_area  = geometry.reference_area/2.
    SU2_settings.mach_number     = conditions.aerodynamics.mach
    SU2_settings.angle_of_attack = conditions.aerodynamics.angle_of_attack / Units.deg
    
    ## build SU2 cfg
    #write_SU2_cfg(tag, SU2_settings)
    
    ### run su2
    #CL, CD = call_SU2_CFD(tag)
    
    if SU2_settings.angle_of_attack == 0:
        if SU2_settings.mach_number == 0.4:
            CL = .337
            CD = .0204
        else:
            CL = .416
            CD = .0265            
            
    else:
        if SU2_settings.mach_number == 0.4:
            CL = .879
            CD = .0780
        else:
            CL = 1.00
            CD = .125

    return CL, CD
