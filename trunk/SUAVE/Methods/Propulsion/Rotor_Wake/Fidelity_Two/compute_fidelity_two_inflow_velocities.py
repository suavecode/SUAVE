## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_Two
# compute_fidelity_two_inflow_velocities.py
#
# Created:  Sep 2022, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_One.compute_fidelity_one_inflow_velocities import compute_fidelity_one_inflow_velocities
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.BET_calculations import compute_inflow_and_tip_loss, compute_airfoil_aerodynamics

# package imports
import numpy as np

def compute_fidelity_two_inflow_velocities(wake, rotor, wake_inputs, conditions):
        

        wake, rotor, interpolatedBoxData = wake.evolve_wake_vortex_distribution(rotor,conditions)
        
        # compute wake-induced velocities
        va, vt = compute_fidelity_one_inflow_velocities(wake,rotor,wake_inputs.ctrl_pts)   
            
        # Update the vortex strengths of each vortex ring accordingly    
        Ua = wake_inputs.velocity_axial
        Ut = wake_inputs.velocity_tangential
        r  = wake_inputs.radius_distribution
        
        R  = rotor.tip_radius
        Rh = rotor.hub_radius
        B  = rotor.number_of_blades    
        
        beta = wake_inputs.twist_distribution
        c = wake_inputs.chord_distribution
        a = wake_inputs.speed_of_sounds
        nu = wake_inputs.dynamic_viscosities
        ctrl_pts = wake_inputs.ctrl_pts
        a_geo = rotor.airfoil_geometry
        a_loc = rotor.airfoil_polar_stations
        cl_sur = rotor.airfoil_cl_surrogates
        cd_sur = rotor.airfoil_cd_surrogates
        Nr = len(rotor.radius_distribution)
        Na = rotor.number_azimuthal_stations
        tc = rotor.thickness_to_chord
        
        # blade velocities
        Wa   = va + Ua
        Wt   = Ut - vt
        
        # blade forces
        lamdaw, F, _ = compute_inflow_and_tip_loss(r,R,Rh,Wa,Wt,B)
        Cl, Cdval, alpha, Ma, W = compute_airfoil_aerodynamics(beta,c,r,R,B,F,Wa,Wt,a,nu,a_loc,a_geo,cl_sur,cd_sur,ctrl_pts,Nr,Na,tc,use_2d_analysis=True)

        # circulation at the blade        
        Gamma = 0.5*W*c*Cl*F     

        gfit = np.zeros_like(r)
        for cpt in range(ctrl_pts):
                # FILTER OUTLIER DATA
                for a in range(Na):
                        gPoly = np.poly1d(np.polyfit(r[0,:,0], Gamma[cpt,:,a], 4))
                        gfit[cpt,:,a] = F[cpt,:,a]*gPoly(r[0,:,0])        
                    
        rotor.outputs.disc_circulation = gfit
        
        gamma_new = (gfit[:,:-1,:] + gfit[:,1:,:])*0.5  # [control points, Nr-1, Na ] one less radial station because ring
        
        m=len(rotor.outputs.omega)
        nts = Na*wake.wake_settings.number_rotor_rotations
        num       = Na//B
        time_idx  = np.arange(nts)
        Gamma     = np.zeros((Na,m,B,Nr-1,nts))
        
        # generate Gamma for each start angle
        for ito in range(Na):
                t_idx     = np.atleast_2d(time_idx).T 
                B_idx     = np.arange(B) 
                B_loc     = (ito + B_idx*num - t_idx )%Na 
                Gamma1    = gamma_new[:,:,B_loc]  
                Gamma1    = Gamma1.transpose(0,3,1,2) 
                Gamma[ito,:,:,:,:] = Gamma1
            
        wake.vortex_distribution.reshaped_wake.GAMMA[:,:,0:B,:,:] = wake.vortex_distribution.reshaped_wake.GAMMA[:,:,0:B,:,:] + 0.5*(Gamma  - wake.vortex_distribution.reshaped_wake.GAMMA[:,:,0:B,:,:])
    
        mat6_size = (Na,m,nts*B*(Nr-1))         
    
        wake.vortex_distribution.GAMMA  =  np.reshape(wake.vortex_distribution.reshaped_wake.GAMMA,mat6_size)
        # recompute with new circulations
        va, vt = compute_fidelity_one_inflow_velocities(wake,rotor,wake_inputs.ctrl_pts)     
        
        return va, vt, rotor