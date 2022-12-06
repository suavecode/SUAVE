## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
# compute_interpolated_velocity_field.py
# 
# Created:  Aug 2022, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np
import copy
from scipy.interpolate import RegularGridInterpolator
from SUAVE.Core import Data
from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_One.compute_wake_induced_velocity import compute_wake_induced_velocity
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_wing_induced_velocity import compute_wing_induced_velocity
import multiprocessing
#import threading
from multiprocessing.pool import Pool
import time

def compute_interpolated_velocity_field(WD_network, rotor, conditions, VD=None, dL=0.025, factor=0.):
    """
    Inputs
       WD            - Rotor wake vortex distribution
       VD            - External vortex distribution (ie. from lifting surface VLM)
       dL            - characteristic length of box size
       factor        - scaling factor to extend box region
    Outputs
       fun_V_induced - function of interpolated velocity based on (x,y,z) position
    
    """
           
    #--------------------------------------------------------------------------------------------
    # Step 1a: Generate coarse grid for external boundaries
    #--------------------------------------------------------------------------------------------   
    R = rotor.tip_radius
    rotor_WD = rotor.Wake.vortex_distribution
    
    # find min/max dimensions of this rotor wake's filaments (rotor frame)
    Xmin = np.min([rotor_WD.XA1, rotor_WD.XA2, rotor_WD.XB1, rotor_WD.XB2]) - (factor * R)
    Xmax = np.max([rotor_WD.XA1, rotor_WD.XA2, rotor_WD.XB1, rotor_WD.XB2]) + (factor * R)
    Ymin = np.min([rotor_WD.YA1, rotor_WD.YA2, rotor_WD.YB1, rotor_WD.YB2]) - (factor * R)
    Ymax = np.max([rotor_WD.YA1, rotor_WD.YA2, rotor_WD.YB1, rotor_WD.YB2]) + (factor * R)
    Zmin = np.min([rotor_WD.ZA1, rotor_WD.ZA2, rotor_WD.ZB1, rotor_WD.ZB2]) - (factor * R)
    Zmax = np.max([rotor_WD.ZA1, rotor_WD.ZA2, rotor_WD.ZB1, rotor_WD.ZB2]) + (factor * R)
    
    Nx = round( (Xmax-Xmin) / dL )
    Ny = round( (Ymax-Ymin) / dL )
    Nz = round( (Zmax-Zmin) / dL )
    
    Xouter = np.linspace(Xmin, Xmax, max(2,Nx)) 
    Youter = np.linspace(Ymin, Ymax, max(2,Ny))
    Zouter = np.linspace(Zmin, Zmax, max(2,Nz))
    
    Xp, Yp, Zp = np.meshgrid(Xouter, Youter, Zouter, indexing='ij')
    
    # stack grid points
    Xstacked = np.reshape(Xp, np.size(Xp))
    Ystacked = np.reshape(Yp, np.size(Yp))
    Zstacked = np.reshape(Zp, np.size(Zp))  
    
    # split into multiple computations for reduced memory requirements and runtime
    #nevals = 6 # Number of jobs to split the computation into
    #maxPts = int(np.ceil(len(Xstacked)/nevals))
    maxPts = 100
    nevals = int(np.ceil(len(Xstacked) / maxPts))
    V_ind_network = np.zeros((1, len(Xstacked), 3))
    Vind_ext = np.zeros((1, len(Xstacked), 3))

    sizeRef = np.size(WD_network[list(WD_network.keys())[0]].XA1)
    
    
    #if sizeRef > 1: #2e5:
        
    with Pool() as pool:
        args = [(i, nevals, maxPts, Xstacked, Ystacked, Zstacked, WD_network,VD, V_ind_network, Vind_ext) for i in range(nevals)]
        
        # execute tasks in order
        i=0
        for result in pool.starmap(multiprocessing_function, args):
            iStart = i*maxPts
            if i == nevals-1:
                iEnd = len(Xstacked)
            else:
                iEnd = (i+1)*maxPts 
                
            # save results
            Vind_ext[:,iStart:iEnd,:] = result[0]
            V_ind_network[:,iStart:iEnd,:] = result[1]
            i = i +1
        
      
    
    # Update induced velocities to appropriate shape
    Vind = np.zeros(np.append(np.shape(Xp), 3))
    Vind[:,:,:,0] = np.reshape(V_ind_network[0,:,0], np.shape(Xp))
    Vind[:,:,:,1] = np.reshape(V_ind_network[0,:,1], np.shape(Xp))
    Vind[:,:,:,2] = np.reshape(V_ind_network[0,:,2], np.shape(Xp))

    V_ind_ext = np.zeros(np.append(np.shape(Xp), 3))
    V_ind_ext[:,:,:,0] = np.reshape(Vind_ext[0,:,0], np.shape(Xp))
    V_ind_ext[:,:,:,1] = np.reshape(Vind_ext[0,:,1], np.shape(Xp))
    V_ind_ext[:,:,:,2] = np.reshape(Vind_ext[0,:,2], np.shape(Xp))    
    
    
    # store box data
    interpolatedBoxData = Data()
    interpolatedBoxData.N_width  = len(Xp[:,0,0]) # x-direction (rotor frame)
    interpolatedBoxData.N_depth  = len(Xp[0,:,0]) # y-direction (rotor frame)
    interpolatedBoxData.N_height = len(Xp[0,0,:]) # z-direction (rotor frame)
    interpolatedBoxData.Position = np.transpose(np.array([Xp, Yp, Zp]), (1,2,3,0))
    interpolatedBoxData.Velocity = Vind
    
    interpolatedBoxData.Dimensions = Data()
    interpolatedBoxData.Dimensions.Xmin = Xmin
    interpolatedBoxData.Dimensions.Xmax = Xmax
    interpolatedBoxData.Dimensions.Ymin = Ymin
    interpolatedBoxData.Dimensions.Ymax = Ymax
    interpolatedBoxData.Dimensions.Zmin = Zmin
    interpolatedBoxData.Dimensions.Zmax = Zmax   
    
    #--------------------------------------------------------------------------------------------    
    # Step 1e: Generate function for induced velocity, Vind = f(x,y,z)
    #--------------------------------------------------------------------------------------------
    Vinf = conditions.frames.inertial.velocity_vector
    rot_mat = rotor.body_to_prop_vel()
    vVec = np.matmul(Vinf, rot_mat)
    
    V_induced = Vind + V_ind_ext + vVec[0]
    
    #fun_V_induced = RegularGridInterpolator((XouterPadded,YouterPadded,ZouterPadded), V_induced,fill_value=None)
    fun_V_induced = RegularGridInterpolator((Xouter,Youter,Zouter), V_induced,bounds_error=False,fill_value=None, method="linear")
    #fun_Vx_induced = RegularGridInterpolator((Xouter,Youter,Zouter), V_induced[:,:,:,0],bounds_error=False,fill_value=None, method="cubic")#method="linear")
    #fun_Vy_induced = RegularGridInterpolator((Xouter,Youter,Zouter), V_induced[:,:,:,1],bounds_error=False,fill_value=None, method="cubic")#method="linear")
    #fun_Vz_induced = RegularGridInterpolator((Xouter,Youter,Zouter), V_induced[:,:,:,2],bounds_error=False,fill_value=None, method="cubic")#method="linear")
    
    #fun_V_induced = Data()
    #fun_V_induced.fun_Vx_induced = fun_Vx_induced
    #fun_V_induced.fun_Vy_induced = fun_Vy_induced
    #fun_V_induced.fun_Vz_induced = fun_Vz_induced
    return fun_V_induced, interpolatedBoxData

def multiprocessing_function(i, nevals, maxPts, Xstacked, Ystacked, Zstacked, WD_network,VD, V_ind_network, Vind_ext):
    t0=time.time()
    iStart = i*maxPts
    if i == nevals-1:
        iEnd = len(Xstacked)
    else:
        iEnd = (i+1)*maxPts
    
    GridPoints = Data()
    GridPoints.XC = Xstacked[iStart:iEnd]
    GridPoints.YC = Ystacked[iStart:iEnd]
    GridPoints.ZC = Zstacked[iStart:iEnd]
    GridPoints.n_cp = len(Xstacked[iStart:iEnd])   
    
    #--------------------------------------------------------------------------------------------
    # Step 1b: Compute induced velocities from each wake at these grid points
    #--------------------------------------------------------------------------------------------
    for WD in WD_network:
        if np.size(WD.XA1) != 0:
            # wake has begun shedding, add the influence from shed wake panels
            V_ind_network[:,iStart:iEnd,:] += compute_wake_induced_velocity(WD, GridPoints, cpts=1)
    
    
    #--------------------------------------------------------------------------------------------    
    # Step 1c: Compute induced velocities at each evaluation point due to external VD
    #--------------------------------------------------------------------------------------------
    if VD is not None:   
        VD_temp = copy.deepcopy(VD)
        VD_temp.XC = GridPoints.XC
        VD_temp.YC = GridPoints.YC
        VD_temp.ZC = GridPoints.ZC
        VD_temp.n_cp = len(GridPoints.ZC)
        C_mn, _, _, _ = compute_wing_induced_velocity(VD_temp, mach=np.array([0]))
        
        Vext_x = (C_mn[:,:,:,0]@VD.gamma.T)
        Vext_y = (C_mn[:,:,:,1]@VD.gamma.T)
        Vext_z = (C_mn[:,:,:,2]@VD.gamma.T)
        Vind_ext[:,iStart:iEnd,:] = np.concatenate([Vext_x, Vext_y, Vext_z],axis=2)
        
    else:
        Vind_ext = np.zeros_like(V_ind_network)  
    
    eval_time = time.time()-t0
    print("\tEvaluation %d out of %d completed in %f seconds" % (i+1, nevals, eval_time))

    return Vind_ext[:,iStart:iEnd,:], V_ind_network[:,iStart:iEnd,:]