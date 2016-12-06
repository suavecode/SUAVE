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

import pyKriging
from pyKriging.krige import kriging

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
        self.settings.parallel         = False
        self.settings.processors       = 1

        # conditions table, used for surrogate model training
        self.training = Data()        
        self.training.angle_of_attack  = np.array([-2.,0,2.]) * Units.deg
        self.training.Mach             = np.array([0.3,0.8,2.])
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
        #drag_model = surrogates.drag_coefficient
        
        ## inviscid lift for Clough interp
        #inviscid_lift                                              = lift_model(AoA,mach)
        # for Kriging
        data_len = len(AoA)
        inviscid_lift = np.zeros([data_len,1])
        for ii,_ in enumerate(AoA):
            inviscid_lift[ii] = lift_model.predict([AoA[ii][0],mach[ii][0]])
        conditions.aerodynamics.lift_breakdown.inviscid_wings_lift = inviscid_lift
        state.conditions.aerodynamics.lift_coefficient             = inviscid_lift
        state.conditions.aerodynamics.lift_breakdown.compressible_wings = inviscid_lift
        
        ## inviscid drag for Clough interp
        #inviscid_drag                                              = drag_model(AoA,mach)
        inviscid_drag = np.zeros([data_len,1])
        #for ii,_ in enumerate(AoA):
            #inviscid_drag[ii] = drag_model.predict([AoA[ii][0],mach[ii][0]])        
        #state.conditions.aerodynamics.inviscid_drag_coefficient    = inviscid_drag

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
        #CD_data   = training.coefficients[:,1]
        xy        = training.grid_points 
        
        # Kriging -------
        
        cl_surrogate = kriging(xy, CL_data)
        cl_surrogate.train()
        #cd_surrogate = kriging(xy, CD_data)
        #cd_surrogate.train()        
        
        # ------
        
        ## rectbivar specific -----
        
        #CL_mesh = CL_data.reshape([2,3])
        #CD_mesh = CD_data.reshape([2,3])
        
        #cl_surrogate = sp.interpolate.RectBivariateSpline(mach_data,AoA_data,CL_mesh,bbox=[.25,.95,-4*Units.deg,6*Units.deg],ky=2,kx=1)
        #cd_surrogate = sp.interpolate.RectBivariateSpline(mach_data,AoA_data,CD_mesh,bbox=[.25,.95,-4*Units.deg,6*Units.deg],ky=2,kx=1)         
        
        ## ----
        
        # learn the model
        #cl_surrogate = sp.interpolate.CloughTocher2DInterpolator(xy,CL_data,fill_value=np.nan)
        #cd_surrogate = sp.interpolate.CloughTocher2DInterpolator(xy,CD_data,fill_value=np.nan)

        self.surrogates.lift_coefficient = cl_surrogate
        #self.surrogates.drag_coefficient = cd_surrogate
        
        import pylab as plt
        #fig = plt.figure('Surrogate Plot')

        
        AoA_points = np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5])*Units.deg
        mach_points = np.array([.35,.45,.55,.65,.75,.8])
        
        AoA_mesh,mach_mesh = np.meshgrid(AoA_points,mach_points)
        
        CL_sur = np.zeros(np.shape(AoA_mesh))
        #CD_sur = np.zeros(np.shape(AoA_mesh))        
        
        for jj in range(len(AoA_points)):
            for ii in range(len(mach_points)):
                CL_sur[ii,jj] = cl_surrogate.predict([AoA_mesh[ii,jj],mach_mesh[ii,jj]])
                #CD_sur[ii,jj] = cd_surrogate.predict([AoA_mesh[ii,jj],mach_mesh[ii,jj]])
        
        #axes = fig.add_subplot(2,1,1)
        #plt.contourf(AoA_mesh/Units.deg,mach_mesh,cl_surrogate(AoA_mesh,mach_mesh))
        #plt.colorbar()
        #plt.xlabel('Angle of Attack (deg)')
        #plt.ylabel('Mach Number')
        #plt.show()
        
        fig = plt.figure('CL - CD Surrogate Plot')    
        axes = fig.add_subplot(2,1,1)
        plt.contourf(AoA_mesh/Units.deg,mach_mesh,CL_sur,levels=None)
        plt.colorbar()
        plt.xlabel('Angle of Attack (deg)')
        plt.ylabel('Mach Number')
        
        #axes = fig.add_subplot(2,1,2)
        #plt.contourf(AoA_mesh/Units.deg,mach_mesh,CD_sur,levels=None)
        #plt.colorbar()
        #plt.xlabel('Angle of Attack (deg)')
        #plt.ylabel('Mach Number')   
        
        #plt.show()

        return



# ----------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------

def call_SU2(conditions,settings,geometry):
    """ calculate total vehicle lift coefficient by SU2
    """

    half_mesh_flag = settings.half_mesh_flag
    tag            = geometry.tag
    parallel       = settings.parallel
    processors     = settings.processors 
    
    SU2_settings = Data()
    if half_mesh_flag == False:
        SU2_settings.reference_area  = geometry.reference_area
    else:
        SU2_settings.reference_area  = geometry.reference_area/2.
    SU2_settings.mach_number     = conditions.aerodynamics.mach
    SU2_settings.angle_of_attack = conditions.aerodynamics.angle_of_attack / Units.deg
    
    ## build SU2 cfg
    write_SU2_cfg(tag, SU2_settings)
    
    # run su2
    CL, CD = call_SU2_CFD(tag,parallel,processors)
    
    ## Results from unrefined half mesh
    
    #if SU2_settings.angle_of_attack == -2:
        #if SU2_settings.mach_number == 0.3:
            #CL = 0.101395
            #CD = 0.0154264
        #else:
            #CL = 0.15306
            #CD = 0.0166416
            
    #elif SU2_settings.angle_of_attack == 0:
        #if SU2_settings.mach_number == 0.3:
            #CL = 0.329861
            #CD = 0.0216294
        #else:
            #CL = 0.454384
            #CD = 0.0319381
            
    #else:
        #if SU2_settings.mach_number == 0.3:
            #CL = 0.873174
            #CD = 0.0751625
        #else:
            #CL = 1.08189
            #CD = 0.152071  
            
    #Results from unrefined half mesh
    
    #if SU2_settings.angle_of_attack == -2:
        #if SU2_settings.mach_number == 0.3:
            #CL = 0.104922
            #CD = 0.00511061
        #else:
            #CL = 0.155651
            #CD = 0.0113375
            
    #elif SU2_settings.angle_of_attack == 0:
        #if SU2_settings.mach_number == 0.3:
            #CL = 0.336125
            #CD = 0.00974673
        #else:
            #CL = 0.461421
            #CD = 0.0251625
            
    #else:
        #if SU2_settings.mach_number == 0.3:
            #CL = 0.890109
            #CD = 0.0513007
        #else:
            #CL = 1.09259
            #CD = 0.138807     
            

    return CL, CD
