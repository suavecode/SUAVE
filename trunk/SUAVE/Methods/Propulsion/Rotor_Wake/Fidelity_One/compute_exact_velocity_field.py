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



def compute_exact_velocity_field(WD, newWD, rotor, conditions, VD=None):
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
    
    #--------------------------------------------------------------------------------------------
    # Step 1a: Generate coarse grid for external boundaries
    #--------------------------------------------------------------------------------------------   
    stack_size = np.size(newWD.XA1)
    # find min/max dimensions of rotor wake filaments (rotor frame)
    Xvals = np.array([np.reshape(newWD.XA1, stack_size), np.reshape(newWD.XA2, stack_size), np.reshape(newWD.XB1, stack_size), np.reshape(newWD.XB2, stack_size)])
    Yvals = np.array([np.reshape(newWD.YA1, stack_size), np.reshape(newWD.YA2, stack_size), np.reshape(newWD.YB1, stack_size), np.reshape(newWD.YB2, stack_size)])
    Zvals = np.array([np.reshape(newWD.ZA1, stack_size), np.reshape(newWD.ZA2, stack_size), np.reshape(newWD.ZB1, stack_size), np.reshape(newWD.ZB2, stack_size)])
    
    # stack grid points
    Xstacked = np.reshape(Xvals, np.size(Xvals))
    Ystacked = np.reshape(Yvals, np.size(Yvals))
    Zstacked = np.reshape(Zvals, np.size(Zvals))
    
    GridPoints = Data()
    GridPoints.XC = Xstacked
    GridPoints.YC = Ystacked
    GridPoints.ZC = Zstacked
    GridPoints.n_cp = len(Xstacked)    

    #--------------------------------------------------------------------------------------------
    # Step 1b: Compute induced velocities at these grid points
    #--------------------------------------------------------------------------------------------
    V_ind_self = compute_wake_induced_velocity(WD, GridPoints, cpts=1)
    
    Vind = np.zeros(np.append(np.shape(Xvals), 3))
    Vind[:,:,:,0] = np.reshape(V_ind_self[0,:,0], np.shape(Xvals))
    Vind[:,:,:,1] = np.reshape(V_ind_self[0,:,1], np.shape(Xvals))
    Vind[:,:,:,2] = np.reshape(V_ind_self[0,:,2], np.shape(Xvals))

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
        V_ind_ext = np.zeros_like(Vind)
        V_ind_ext[:,:,:,0] = np.reshape( (C_mn[:,:,:,0]@VD.gamma.T)[0,:,0] , np.shape(V_ind_ext[:,:,:,0]) )
        V_ind_ext[:,:,:,1] = np.reshape( (C_mn[:,:,:,1]@VD.gamma.T)[0,:,0] , np.shape(V_ind_ext[:,:,:,0]) )
        V_ind_ext[:,:,:,2] = np.reshape( (C_mn[:,:,:,2]@VD.gamma.T)[0,:,0] , np.shape(V_ind_ext[:,:,:,0]) )
    else:
        V_ind_ext = 0
        

    
    #--------------------------------------------------------------------------------------------    
    # Step 1d: Generate function for induced velocity, Vind = f(x,y,z)
    #--------------------------------------------------------------------------------------------
    Vinf = conditions.frames.inertial.velocity_vector
    rot_mat = rotor.body_to_prop_vel()
    vVec = np.matmul(Vinf, rot_mat)
    
    V_induced = Vind + V_ind_ext + vVec[0]
    
    V_nodes = Data()
    V_nodes.A1 = V_induced[0,:]

    return V_induced