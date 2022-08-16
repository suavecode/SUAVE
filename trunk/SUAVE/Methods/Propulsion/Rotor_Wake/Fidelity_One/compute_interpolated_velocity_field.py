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



def compute_interpolated_velocity_field(WD, conditions, VD=None, dL=0.05, factor=1.5):
    """
    Inputs
       WD            - Rotor wake vortex distribution
       VD            - External vortex distribution
       dL            - characteristic length of box size
       factor        - scaling factor to extend box region
    Outputs
       fun_V_induced - function of interpolated velocity based on (x,y,z) position
    
    """
    
    #--------------------------------------------------------------------------------------------
    # Step 1a: Generate coarse grid for external boundaries
    #--------------------------------------------------------------------------------------------   
    # find min/max dimensions of rotor wake filaments (rotor frame)
    Xmin = np.min([WD.XA1, WD.XA2, WD.XB1, WD.XB2])
    Xmax = np.max([WD.XA1, WD.XA2, WD.XB1, WD.XB2])
    Ymin = np.min([WD.YA1, WD.YA2, WD.YB1, WD.YB2])
    Ymax = np.max([WD.YA1, WD.YA2, WD.YB1, WD.YB2])
    Zmin = np.min([WD.ZA1, WD.ZA2, WD.ZB1, WD.ZB2])
    Zmax = np.max([WD.ZA1, WD.ZA2, WD.ZB1, WD.ZB2])
    
    Nx = round( (Xmax-Xmin) / dL )
    Ny = round( (Ymax-Ymin) / dL )
    Nz = round( (Zmax-Zmin) / dL )
    
    Xouter = np.linspace(Xmin, Xmax, Nx) * factor 
    Youter = np.linspace(Ymin, Ymax, Ny) * factor
    Zouter = np.linspace(Zmin, Zmax, Nz) * factor
    
    Xp, Yp, Zp = np.meshgrid(Xouter, Youter, Zouter, indexing='ij')
    
    # stack grid points
    Xstacked = np.reshape(Xp, np.size(Xp))
    Ystacked = np.reshape(Yp, np.size(Yp))
    Zstacked = np.reshape(Zp, np.size(Zp))
    
    GridPoints = Data()
    GridPoints.XC = Xstacked
    GridPoints.YC = Ystacked
    GridPoints.ZC = Zstacked
    GridPoints.n_cp = len(Xstacked)    

    #--------------------------------------------------------------------------------------------
    # Step 1b: Compute induced velocities at these grid points
    #--------------------------------------------------------------------------------------------
    V_ind_self = compute_wake_induced_velocity(WD, GridPoints, cpts=1)
    
    Vind = np.zeros(np.append(np.shape(Xp), 3))
    Vind[:,:,:,0] = np.reshape(V_ind_self[0,:,0], np.shape(Xp))
    Vind[:,:,:,1] = np.reshape(V_ind_self[0,:,1], np.shape(Xp))
    Vind[:,:,:,2] = np.reshape(V_ind_self[0,:,2], np.shape(Xp))

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
        
    interpolatedBoxData = Data()
    interpolatedBoxData.N_width  = len(Xp[:,0,0]) # x-direction (rotor frame)
    interpolatedBoxData.N_depth  = len(Xp[0,:,0]) # y-direction (rotor frame)
    interpolatedBoxData.N_height = len(Xp[0,0,:]) # z-direction (rotor frame)
    interpolatedBoxData.Position = np.transpose(np.array([Xp, Yp, Zp]), (1,2,3,0))
    interpolatedBoxData.Velocity = Vind
    
    #--------------------------------------------------------------------------------------------    
    # Step 1d: Generate function for induced velocity, Vind = f(x,y,z)
    #--------------------------------------------------------------------------------------------
    Vinf = conditions.frames.inertial.velocity_vector
    V_induced = Vind + V_ind_ext + Vinf
    
    fun_V_induced = RegularGridInterpolator((Xouter,Youter,Zouter), V_induced)
    
    return fun_V_induced, interpolatedBoxData