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



def compute_exact_velocity_field(WD_network, rotor, conditions, VD_VLM=None, GridPointsFull=None):
    """
    Inputs
       WD            - Rotor wake vortex distribution
       newWD         - New points at which to compute the velocity
       VD            - External vortex distribution
       dL            - characteristic length of box size
       factor        - scaling factor to extend box region
    Outputs
       fun_V_induced - function of interpolated velocity based on (x,y,z) position
    
    """
     
    if GridPointsFull is None:
        #--------------------------------------------------------------------------------------------
        # Step 1a: Generate grid points for evaluation
        #--------------------------------------------------------------------------------------------  
        stack_size = np.size(WD.XA1)
        # find min/max dimensions of rotor wake filaments (rotor frame)
        Xvals = np.array([np.reshape(WD.XA1, stack_size), np.reshape(WD.XA2, stack_size), np.reshape(WD.XB1, stack_size), np.reshape(WD.XB2, stack_size)])
        Yvals = np.array([np.reshape(WD.YA1, stack_size), np.reshape(WD.YA2, stack_size), np.reshape(WD.YB1, stack_size), np.reshape(WD.YB2, stack_size)])
        Zvals = np.array([np.reshape(WD.ZA1, stack_size), np.reshape(WD.ZA2, stack_size), np.reshape(WD.ZB1, stack_size), np.reshape(WD.ZB2, stack_size)])
        
        # stack grid points
        Xstacked = np.reshape(Xvals, np.size(Xvals))
        Ystacked = np.reshape(Yvals, np.size(Yvals))
        Zstacked = np.reshape(Zvals, np.size(Zvals))
        
        GridPointsFull = Data()
        GridPointsFull.XC = Xstacked
        GridPointsFull.YC = Ystacked
        GridPointsFull.ZC = Zstacked
        GridPointsFull.n_cp = len(Xstacked)    

    #--------------------------------------------------------------------------------------------
    # Step 1b: Compute induced velocities at these grid points
    #--------------------------------------------------------------------------------------------
    
    # split into multiple computations for reduced memory requirements and runtime
    maxPts = 1000
    nevals = int(np.ceil(len(GridPointsFull.XC) / maxPts))
    V_ind_self = np.zeros((1, len(GridPointsFull.XC), 3))
    import time
    t0 = time.time()
    for i in range(nevals):
        iStart = i*maxPts
        if i == nevals-1:
            iEnd = len(GridPointsFull.XC)
        else:
            iEnd = (i+1)*maxPts
        
        GridPoints = Data()
        GridPoints.XC = GridPointsFull.XC[iStart:iEnd]
        GridPoints.YC = GridPointsFull.YC[iStart:iEnd]
        GridPoints.ZC = GridPointsFull.ZC[iStart:iEnd]
        GridPoints.n_cp = len(GridPointsFull.XC[iStart:iEnd])   
        
        #--------------------------------------------------------------------------------------------
        # Step 1b: Compute induced velocities from each wake at these grid points
        #--------------------------------------------------------------------------------------------
        for WD in WD_network:
            if np.size(WD.XA1) != 0:
                # wake has begun shedding, add the influence from shed wake panels
                V_ind_self[:,iStart:iEnd,:] += compute_wake_induced_velocity(WD, GridPoints, cpts=1)
                
                
    print("\t" + str(time.time() - t0))
    #Vind = np.zeros(np.append(np.shape(Xvals), 3))
    #Vind[:,:,0] = np.reshape(V_ind_self[0,:,0], np.shape(Xvals))
    #Vind[:,:,1] = np.reshape(V_ind_self[0,:,1], np.shape(Xvals))
    #Vind[:,:,2] = np.reshape(V_ind_self[0,:,2], np.shape(Xvals))

    #--------------------------------------------------------------------------------------------    
    # Step 1c: Compute induced velocities at each evaluation point due to external VD
    #--------------------------------------------------------------------------------------------
    if VD_VLM is not None:   
        VD_temp = copy.deepcopy(VD_VLM)
        VD_temp.XC = GridPointsFull.XC
        VD_temp.YC = GridPointsFull.YC
        VD_temp.ZC = GridPointsFull.ZC
        VD_temp.n_cp = len(GridPointsFull.ZC)
        C_mn, _, _, _ = compute_wing_induced_velocity(VD_temp, mach=np.array([0]))
        V_ind_ext = np.zeros_like(V_ind_self)
        V_ind_ext[:,:,0] = np.reshape( (C_mn[:,:,:,0]@VD_VLM.gamma.T)[0,:,0] , np.shape(V_ind_ext[:,:,0]) )
        V_ind_ext[:,:,1] = np.reshape( (C_mn[:,:,:,1]@VD_VLM.gamma.T)[0,:,0] , np.shape(V_ind_ext[:,:,0]) )
        V_ind_ext[:,:,2] = np.reshape( (C_mn[:,:,:,2]@VD_VLM.gamma.T)[0,:,0] , np.shape(V_ind_ext[:,:,0]) )
    else:
        V_ind_ext = 0
        

    
    #--------------------------------------------------------------------------------------------    
    # Step 1d: Generate function for induced velocity, Vind = f(x,y,z)
    #--------------------------------------------------------------------------------------------
    Vinf = conditions.frames.inertial.velocity_vector
    rot_mat = rotor.body_to_prop_vel()
    vVec = np.matmul(Vinf, rot_mat)
    
    V_induced = V_ind_self + V_ind_ext + vVec[0]
    
    #V_nodes = Data()
    #V_nodes.A1 = V_induced[0,:]

    return V_induced