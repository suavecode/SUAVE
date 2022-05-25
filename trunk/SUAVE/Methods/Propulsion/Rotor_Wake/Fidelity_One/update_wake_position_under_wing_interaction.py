## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
# update_wake_position_under_component_interaction.py
# 
# Created:  May 2022, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_wing_induced_velocity import compute_wing_induced_velocity

# package imports
import numpy as np 

## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
def update_wake_position_under_wing_interaction(wake, rotor, wing, vehicle, VD_wing, wing_vlm_outputs):  
    """
    Takes the vortex distribution of another aircraft component and evolves the rotor
    wake's vortex filament positions to account for the presence of the nearby component interaction.
    
    This is an approximation method intended to account for first order effects of interactions that
    will change the rotor wake shape. 
    
    Assumptions:
    N/A
    
    Source:
    N/A
    
    Inputs:
       wake           - a fidelity one rotor wake
       VD_component   - Vortex distribution associated with the component of interest
    
    Outputs:
       VD_wake        - Updated vortex distribution of the rotor wake due to component interaction
        
    """    

    # -------------------------------------------------------------------------------------------    
    #        Loop over each time step to update the rotor wake shape
    # -------------------------------------------------------------------------------------------    
    nts = wake.wake_settings.number_steps_per_rotation * wake.wake_settings.number_rotor_rotations
    for t in range(nts):
        # ------------------------------------------------------------------------------------------- 
        # Compute the velocity induced by VD_component at the trailing edge panels of row t
        # ------------------------------------------------------------------------------------------- 
        panels = wake.vortex_distribution
        
        p_shape = np.shape(panels.reshaped_wake.XA2[:,:,:,:,t])
        p_size = np.size(panels.reshaped_wake.XA2[:,:,:,:,t])
        
        # Update the collocation points to compute the wing induced velocities at the wake panel TE
        stacked_XA2 = np.reshape(panels.reshaped_wake.XA2[:,:,:,:,t], p_size)
        stacked_XB2 = np.reshape(panels.reshaped_wake.XB2[:,:,:,:,t], p_size)
        stacked_YA2 = np.reshape(panels.reshaped_wake.YA2[:,:,:,:,t], p_size)
        stacked_YB2 = np.reshape(panels.reshaped_wake.YB2[:,:,:,:,t], p_size)  
        stacked_ZA2 = np.reshape(panels.reshaped_wake.ZA2[:,:,:,:,t], p_size)
        stacked_ZB2 = np.reshape(panels.reshaped_wake.ZB2[:,:,:,:,t], p_size)        
        
        XC          = np.append(stacked_XA2,stacked_XB2)
        YC          = np.append(stacked_YA2,stacked_YB2)
        ZC          = np.append(stacked_ZA2,stacked_ZB2)
        
        VD_wing.XC  = XC     # (Na, ctrl_pts, B, Nr, t)
        VD_wing.YC  = YC
        VD_wing.ZC  = ZC
        
        # Compute the wing induced velocities at these control points
        gammaT = wing_vlm_outputs.gamma.T
        C_mn, s, RFLAG, EW = compute_wing_induced_velocity(VD_wing,mach=[[0.]],compute_EW=False)
        u_inviscid    = (C_mn[:,:,:,0]@gammaT)[0,:,0]
        v_inviscid    = (C_mn[:,:,:,1]@gammaT)[0,:,0]
        w_inviscid    = (C_mn[:,:,:,2]@gammaT)[0,:,0]  
        
        u_XA2 = np.reshape( u_inviscid[0:p_size] , p_shape)
        u_XB2 = np.reshape( u_inviscid[p_size:]  , p_shape)
        v_XA2 = np.reshape( v_inviscid[0:p_size] , p_shape)
        v_XB2 = np.reshape( v_inviscid[p_size:]  , p_shape)  
        w_XA2 = np.reshape( w_inviscid[0:p_size] , p_shape)
        w_XB2 = np.reshape( w_inviscid[p_size:]  , p_shape)        

        # -------------------------------------------------------------------------------------------         
        # Update the position of all panels in rotor wake
        # ------------------------------------------------------------------------------------------- 
        Na = p_shape[0]
        omega = rotor.inputs.omega
        
        # Compute blade angles starting from each of Na azimuthal stations, shape: (Na,B)
        dazi = 2*np.pi / Na
        dt   = dazi/omega[0][0]
        ts   = np.linspace(t,dt*(nts-1),nts-t)
        dts   = np.ones_like(panels.reshaped_wake.XA2[:,:,:,:,t:]) *ts*dt
        
        # Rotate trailing edge points of this row's panels according to the new induced velocities
        # Translate all panels for now (rotate later)
        wake.vortex_distribution.reshaped_wake.XA2[:,:,:,:,t:] += np.repeat(u_XA2[:,:,:,:,None],nts-t,axis=4) * dts
        wake.vortex_distribution.reshaped_wake.YA2[:,:,:,:,t:] += np.repeat(v_XA2[:,:,:,:,None],nts-t,axis=4) * dts
        wake.vortex_distribution.reshaped_wake.ZA2[:,:,:,:,t:] += np.repeat(w_XA2[:,:,:,:,None],nts-t,axis=4) * dts
        wake.vortex_distribution.reshaped_wake.XB2[:,:,:,:,t:] += np.repeat(u_XB2[:,:,:,:,None],nts-t,axis=4) * dts
        wake.vortex_distribution.reshaped_wake.YB2[:,:,:,:,t:] += np.repeat(v_XB2[:,:,:,:,None],nts-t,axis=4) * dts
        wake.vortex_distribution.reshaped_wake.ZB2[:,:,:,:,t:] += np.repeat(w_XB2[:,:,:,:,None],nts-t,axis=4) * dts
        
        wake.vortex_distribution.XA2 = np.reshape(wake.vortex_distribution.reshaped_wake.XA2, np.shape(wake.vortex_distribution.XA2))
        wake.vortex_distribution.YA2 = np.reshape(wake.vortex_distribution.reshaped_wake.YA2, np.shape(wake.vortex_distribution.YA2))
        wake.vortex_distribution.ZA2 = np.reshape(wake.vortex_distribution.reshaped_wake.ZA2, np.shape(wake.vortex_distribution.ZA2))
        wake.vortex_distribution.XB2 = np.reshape(wake.vortex_distribution.reshaped_wake.XB2, np.shape(wake.vortex_distribution.XB2))
        wake.vortex_distribution.YB2 = np.reshape(wake.vortex_distribution.reshaped_wake.YB2, np.shape(wake.vortex_distribution.YB2))
        wake.vortex_distribution.ZB2 = np.reshape(wake.vortex_distribution.reshaped_wake.ZB2, np.shape(wake.vortex_distribution.ZB2))
        wake.updated = True
        
        # match all leading edge panels to the trailing edge of prior panel
        wake.vortex_distribution.reshaped_wake.XA1[:,:,:,:,1:] = wake.vortex_distribution.reshaped_wake.XA2[:,:,:,:,:-1]
        wake.vortex_distribution.reshaped_wake.YA1[:,:,:,:,1:] = wake.vortex_distribution.reshaped_wake.YA2[:,:,:,:,:-1]
        wake.vortex_distribution.reshaped_wake.ZA1[:,:,:,:,1:] = wake.vortex_distribution.reshaped_wake.ZA2[:,:,:,:,:-1]
        wake.vortex_distribution.reshaped_wake.XB1[:,:,:,:,1:] = wake.vortex_distribution.reshaped_wake.XB2[:,:,:,:,:-1]
        wake.vortex_distribution.reshaped_wake.YB1[:,:,:,:,1:] = wake.vortex_distribution.reshaped_wake.YB2[:,:,:,:,:-1]
        wake.vortex_distribution.reshaped_wake.ZB1[:,:,:,:,1:] = wake.vortex_distribution.reshaped_wake.ZB2[:,:,:,:,:-1]
        
        wake.vortex_distribution.XA1 = np.reshape(wake.vortex_distribution.reshaped_wake.XA1, np.shape(wake.vortex_distribution.XA1))
        wake.vortex_distribution.YA1 = np.reshape(wake.vortex_distribution.reshaped_wake.YA1, np.shape(wake.vortex_distribution.YA1))
        wake.vortex_distribution.ZA1 = np.reshape(wake.vortex_distribution.reshaped_wake.ZA1, np.shape(wake.vortex_distribution.ZA1))
        wake.vortex_distribution.XB1 = np.reshape(wake.vortex_distribution.reshaped_wake.XB1, np.shape(wake.vortex_distribution.XB1))
        wake.vortex_distribution.YB1 = np.reshape(wake.vortex_distribution.reshaped_wake.YB1, np.shape(wake.vortex_distribution.YB1))
        wake.vortex_distribution.ZB1 = np.reshape(wake.vortex_distribution.reshaped_wake.ZB1, np.shape(wake.vortex_distribution.ZB1))       

        # ------------------------------------------------------------------------------------------- 
        # -------------------------------------------------------------------------------------------         
        # store vtks for debugging
        # ------------------------------------------------------------------------------------------- 
        # ------------------------------------------------------------------------------------------- 
        #from SUAVE.Input_Output.VTK.save_vehicle_vtk import save_vehicle_vtks
        #save_vehicle_vtks(vehicle,time_step=t,save_loc='/Users/rerha/Desktop/Wake_Evolution/')
        

        # ------------------------------------------------------------------------------------------- 
        # -------------------------------------------------------------------------------------------     


    return 
