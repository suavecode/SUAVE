# SU2_inviscid.py
#
# Created:  Sep 2016, E. Botero
# Modified: Jan 2017, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
import SUAVE
from SUAVE.Core import Data, Units

# Local imports
from Aerodynamics import Aerodynamics
from SUAVE.Input_Output.SU2.call_SU2_CFD import call_SU2_CFD
from SUAVE.Input_Output.SU2.write_SU2_cfg import write_SU2_cfg

# Package imports
import numpy as np
import time
import pylab as plt
import sklearn
from sklearn import gaussian_process
from sklearn import neighbors
from sklearn import svm

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class SU2_inviscid_Super(Aerodynamics):

    def __defaults__(self):

        self.tag = 'SU2_inviscid'

        self.geometry = Data()
        self.settings = Data()
        self.settings.half_mesh_flag     = True
        self.settings.parallel           = False
        self.settings.processors         = 1
        self.settings.maximum_iterations = 1500

        # Conditions table, used for surrogate model training
        self.training = Data()        
        self.training.angle_of_attack  = np.array([-2.,3.,8.]) * Units.deg
        self.training.Mach             = np.array([0.3,0.7,0.85])
        self.training.lift_coefficient = None
        self.training.drag_coefficient = None
        self.training_file             = None
        
        # Surrogate model
        self.surrogates = Data()
 
        
    def initialize(self):
                   
        # Sample training data
        self.sample_training()
                    
        # Build surrogate
        self.build_surrogate()


    def evaluate(self,state,settings,geometry):

        # Unpack
        surrogates = self.surrogates        
        conditions = state.conditions
        
        mach = conditions.freestream.mach_number
        AoA  = conditions.aerodynamics.angle_of_attack
        lift_model_sub = surrogates.lift_coefficient_subsonic
        lift_model_sup = surrogates.lift_coefficient_supersonic
        drag_model_sub = surrogates.drag_coefficient_subsonic
        drag_model_sup = surrogates.drag_coefficient_supersonic
        
        # Inviscid lift
        data_len = len(AoA)
        inviscid_lift = np.zeros([data_len,1])
        inviscid_drag = np.zeros([data_len,1]) # Inviscid drag, zeros are a placeholder for possible future implementation
        for ii,_ in enumerate(AoA):
            if mach[ii][0] <= 1.:
                inviscid_lift[ii] = lift_model_sub.predict(np.array([AoA[ii][0],mach[ii][0]]))
                inviscid_drag[ii] = drag_model_sub.predict(np.array([AoA[ii][0],mach[ii][0]]))
            else:
                inviscid_lift[ii] = lift_model_sup.predict(np.array([AoA[ii][0],mach[ii][0]]))
                inviscid_drag[ii] = drag_model_sup.predict(np.array([AoA[ii][0],mach[ii][0]]))
        conditions.aerodynamics.lift_breakdown.inviscid_wings_lift = inviscid_lift
        state.conditions.aerodynamics.lift_coefficient             = inviscid_lift
        state.conditions.aerodynamics.lift_breakdown.compressible_wings = inviscid_lift
        
        state.conditions.aerodynamics.drag_breakdown.inviscid        = Data()
        state.conditions.aerodynamics.drag_breakdown.inviscid.total  = inviscid_drag
        
        return inviscid_lift, inviscid_drag


    def sample_training(self):
        
        # Unpack
        geometry = self.geometry
        settings = self.settings
        training = self.training
        
        t_set = self.training_set
        CL = np.zeros(np.shape(t_set))
        CD = np.zeros(np.shape(t_set))
        
        #AoA  = training.angle_of_attack
        #mach = training.Mach 
        #CL   = np.zeros([len(AoA)*len(mach),1])
        #CD   = np.zeros([len(AoA)*len(mach),1])

        # Condition input, local, do not keep (k is used to avoid confusion)
        konditions              = Data()
        konditions.aerodynamics = Data()

        if self.training_file is None:
            # Calculate aerodynamics for table
            #table_size = len(AoA)*len(mach)
            #xy = np.zeros([table_size,2])
            xy = t_set
            #count = 0
            time0 = time.time()
            #for i,_ in enumerate(AoA):
                #for j,_ in enumerate(mach):
                    
                    #xy[count,:] = np.array([AoA[i],mach[j]])
                    ## Set training conditions
                    #konditions.aerodynamics.angle_of_attack = AoA[i]
                    #konditions.aerodynamics.mach            = mach[j]
                    
                    #CL[count],CD[count] = call_SU2(konditions, settings, geometry)
                    #count += 1
            for ii in xrange(np.shape(t_set)[0]):
                konditions.aerodynamics.angle_of_attack = t_set[ii,0]
                konditions.aerodynamics.mach            = t_set[ii,1]
                CL[ii],CD[ii] = call_SU2(konditions, settings, geometry)
            
            time1 = time.time()
            
            print 'The total elapsed time to run SU2: '+ str(time1-time0) + '  Seconds'
        else:
            data_array = np.loadtxt(self.training_file)
            xy         = data_array[:,0:2]
            CL         = data_array[:,2:3]
            CD         = data_array[:,3:4]

        # Save the data
        np.savetxt(geometry.tag+'_data.txt',np.hstack([xy,CL,CD]),fmt='%10.8f',header='AoA Mach CL CD')

        # Store training data
        training.coefficients = np.hstack([CL,CD])
        training.grid_points  = xy
        

        return

    def build_surrogate(self):

        # Unpack data
        training  = self.training
        AoA_data  = training.angle_of_attack
        mach_data = training.Mach
        CL_data   = training.coefficients[:,0]
        CD_data   = training.coefficients[:,1]
        xy        = training.grid_points 
        
        #import pyKriging
        
        # Gaussian Process New
        regr_cl_sup = gaussian_process.GaussianProcess()
        regr_cl_sub = gaussian_process.GaussianProcess()
        cl_surrogate_sup = regr_cl_sup.fit(xy[xy[:,1]>=1.], CL_data[xy[:,1]>=1.])
        cl_surrogate_sub = regr_cl_sub.fit(xy[xy[:,1]<=1.], CL_data[xy[:,1]<=1.])  
        regr_cd_sup = gaussian_process.GaussianProcess()
        regr_cd_sub = gaussian_process.GaussianProcess()
        cd_surrogate_sup = regr_cd_sup.fit(xy[xy[:,1]>=1.], CD_data[xy[:,1]>=1.])
        cd_surrogate_sub = regr_cd_sub.fit(xy[xy[:,1]<=1.], CD_data[xy[:,1]<=1.])        
        
        # Gaussian Process New
        #regr_cl = gaussian_process.GaussianProcessRegressor()
        #regr_cd = gaussian_process.GaussianProcessRegressor()
        #cl_surrogate = regr_cl.fit(xy, CL_data)
        #cd_surrogate = regr_cd.fit(xy, CD_data)  
        
        # KNN
        #regr_cl = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
        #regr_cd = neighbors.KNeighborsRegressor(n_neighbors=1,weights='distance')
        #cl_surrogate = regr_cl.fit(xy, CL_data)
        #cd_surrogate = regr_cd.fit(xy, CD_data)  
        
        # SVR
        #regr_cl = svm.SVR(C=500.)
        #regr_cd = svm.SVR()
        #cl_surrogate = regr_cl.fit(xy, CL_data)
        #cd_surrogate = regr_cd.fit(xy, CD_data)          
        
        
        self.surrogates.lift_coefficient_subsonic = cl_surrogate_sub
        self.surrogates.lift_coefficient_supersonic = cl_surrogate_sup
        self.surrogates.drag_coefficient_subsonic = cd_surrogate_sub
        self.surrogates.drag_coefficient_supersonic = cd_surrogate_sup
         
        
        # Standard supersonic test case
        AoA_points = np.linspace(-1.1,7.1,100)*Units.deg
        mach_points = np.linspace(.2,2.05,100)      
        
        AoA_mesh,mach_mesh = np.meshgrid(AoA_points,mach_points)
        
        CL_sur = np.zeros(np.shape(AoA_mesh))
        CD_sur = np.zeros(np.shape(AoA_mesh))        
        
        for jj in range(len(AoA_points)):
            for ii in range(len(mach_points)):
                if mach_mesh[ii,jj] >= 1. :
                    CL_sur[ii,jj] = cl_surrogate_sup.predict(np.array([AoA_mesh[ii,jj],mach_mesh[ii,jj]]))
                    CD_sur[ii,jj] = cd_surrogate_sup.predict(np.array([AoA_mesh[ii,jj],mach_mesh[ii,jj]]))
                else:
                    CL_sur[ii,jj] = cl_surrogate_sub.predict(np.array([AoA_mesh[ii,jj],mach_mesh[ii,jj]]))
                    CD_sur[ii,jj] = cd_surrogate_sub.predict(np.array([AoA_mesh[ii,jj],mach_mesh[ii,jj]]))
        

        fig = plt.figure('Coefficient of Lift Surrogate Plot')    
        plt_handle = plt.contourf(AoA_mesh/Units.deg,mach_mesh,CL_sur,levels=None)
        #plt.clabel(plt_handle, inline=1, fontsize=10)
        cbar = plt.colorbar()
        plt.scatter(xy[:,0]/Units.deg,xy[:,1])
        plt.xlabel('Angle of Attack (deg)')
        plt.ylabel('Mach Number')
        cbar.ax.set_ylabel('Coefficient of Lift')

        fig = plt.figure('Coefficient of Drag Surrogate Plot')    
        #levals = [.0,.0025,.005,.0075,.01,.0125,.015,.0175,.02,.0225]
        levals = None
        plt_handle = plt.contourf(AoA_mesh/Units.deg,mach_mesh,CD_sur,levels=levals)
        #plt.clabel(plt_handle, inline=1, fontsize=10)
        cbar = plt.colorbar()
        plt.scatter(xy[:,0]/Units.deg,xy[:,1])
        plt.xlabel('Angle of Attack (deg)')
        plt.ylabel('Mach Number')
        cbar.ax.set_ylabel('Coefficient of Drag (----------NOT USED----------)') 
        
        fig = plt.figure('L/D')   
        #levals=[8,8.5,9,9.5,10,10.5]
        levals=None
        plt_handle = plt.contourf(AoA_mesh/Units.deg,mach_mesh,CL_sur/CD_sur,levels=levals)
        #plt.clabel(plt_handle, inline=1, fontsize=10)
        cbar = plt.colorbar()
        plt.scatter(xy[:,0]/Units.deg,xy[:,1])
        plt.xlabel('Angle of Attack (deg)')
        plt.ylabel('Mach Number')
        cbar.ax.set_ylabel('L/D')         
        
        cl_cruise = cl_surrogate_sup.predict(np.array([2.1*Units.deg,2.0]))
        cd_cruise = cd_surrogate_sup.predict(np.array([2.1*Units.deg,2.0]))        
        print cd_cruise
        
        plt.show() 

        return



# ----------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------

def call_SU2(conditions,settings,geometry):
    """ calculate total vehicle lift coefficient with SU2
    """

    half_mesh_flag = settings.half_mesh_flag
    tag            = geometry.tag
    parallel       = settings.parallel
    processors     = settings.processors 
    iters          = settings.maximum_iterations
    
    SU2_settings = Data()
    if half_mesh_flag == False:
        SU2_settings.reference_area  = geometry.reference_area
    else:
        SU2_settings.reference_area  = geometry.reference_area/2.
    SU2_settings.mach_number     = conditions.aerodynamics.mach
    SU2_settings.angle_of_attack = conditions.aerodynamics.angle_of_attack / Units.deg
    SU2_settings.maximum_iterations = iters
    
    # Build SU2 configuration file
    write_SU2_cfg(tag, SU2_settings)
    
    # Run SU2
    CL, CD = call_SU2_CFD(tag,parallel,processors)
        
    return CL, CD
