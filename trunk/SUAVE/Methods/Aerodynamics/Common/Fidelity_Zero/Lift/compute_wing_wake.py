## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# compute_wing_wake.py
# 
# Created:   April 2021, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import copy
import numpy as np
import pylab as plt
from SUAVE.Core import Data
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.VLM import VLM
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_wing_induced_velocity import compute_wing_induced_velocity
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.generate_wing_wake_grid import generate_wing_wake_grid


def compute_wing_wake(geometry, conditions, x, grid_settings, VLM_settings, viscous_wake=True, plot_grid=False, plot_wake=False):
    """
     Computes the wing-induced velocities at a given x-plane.
     
     Inputs:
        geometry    
             .wings                                                                       [Unitless]
        conditions                                                                        
             .freestream                                                                  [Unitless]
        x                     - streamwise location for evaluating wake                   [m]
        viscous_wake          - flag for using viscous wake correction                    [Boolean]
        grid_settings
             .N
     
     Outputs:
        wing_wake
              .u_velocities   - streamwise induced velocities at control points           [Unitless]
              .v_velocities   - spanwise induced velocities at control points             [Unitless]
              .w_velocities   - induced downwash at control points                        [Unitless]
              .VD             - vortex distribution of wing and control points in wake    [Unitless]
     
    """     
    
    #-------------------------------------------------------------------------
    #          Extract variables:
    #-------------------------------------------------------------------------
    span    = geometry.wings.main_wing.spans.projected
    croot   = geometry.wings.main_wing.chords.root
    ctip    = geometry.wings.main_wing.chords.tip
    x0_wing = geometry.wings.main_wing.origin[0,0]
    aoa     = conditions.aerodynamics.angle_of_attack
    mach    = conditions.freestream.mach_number
    rho     = conditions.freestream.density
    mu      = conditions.freestream.dynamic_viscosity 
    Vv      = conditions.freestream.velocity[0,0]
    nu      = (mu/rho)[0,0]
    H       = grid_settings.height
    L       = grid_settings.length
    hf      = grid_settings.height_fine
    
    # --------------------------------------------------------------------------------
    #          Run the VLM for the given vehicle and conditions 
    # --------------------------------------------------------------------------------
    results = VLM(conditions, VLM_settings, geometry)
    gamma   = results.gamma
    VD      = geometry.vortex_distribution 
    
    # create a deep copy of the vortex distribution
    VD       = copy.deepcopy(VD)
    gammaT   = gamma.T
    
    # ------------------------------------------------------------------------------------
    #          Generate grid points to evaluate wing induced velocities at
    # ------------------------------------------------------------------------------------ 
    grid_points = generate_wing_wake_grid(geometry, H, L, hf, x, plot_grid=plot_grid)
    cp_XC = grid_points.XC
    cp_YC = grid_points.YC
    cp_ZC = grid_points.ZC
    
    VD.XC = cp_XC
    VD.YC = cp_YC
    VD.ZC = cp_ZC  
    
    #----------------------------------------------------------------------------------------------
    # Compute wing induced velocity    
    #----------------------------------------------------------------------------------------------    
    C_mn, _, _, _ = compute_wing_induced_velocity(VD,mach)     
    u_inviscid    = (C_mn[:,:,:,0]@gammaT)[0,:,0]
    v_inviscid    = (C_mn[:,:,:,1]@gammaT)[0,:,0]
    w_inviscid    = (C_mn[:,:,:,2]@gammaT)[0,:,0]     
    
    #----------------------------------------------------------------------------------------------
    # Impart the wake deficit from BL of wing if x is behind the wing
    #----------------------------------------------------------------------------------------------
    Va_deficit = np.zeros_like(VD.YC)
    
    if viscous_wake and (x>=x0_wing):
        # Reynolds number developed at x-plane:
        Rex_prop_plane     = Vv*(x-x0_wing)/nu
        
        # impart viscous wake to grid points within the span of the wing
        y_inside            = abs(VD.YC)<0.5*span
        chord_distribution  = croot - (croot-ctip)*(abs(VD.YC[y_inside])/(0.5*span))
        
        # boundary layer development distance
        x_dev      = (x-x0_wing) * np.ones_like(chord_distribution)
        
        # For turbulent flow
        theta_turb  = 0.036*x_dev/(Rex_prop_plane**(1/5))
        x_theta     = (x_dev-chord_distribution)/theta_turb

        # axial velocity deficit due to turbulent BL from the wing (correlation from Ramaprian et al.)
        W0  = Vv/np.sqrt(4*np.pi*0.032*x_theta)
        b   = 2*theta_turb*np.sqrt(16*0.032*np.log(2)*x_theta)
        Va_deficit[y_inside] = W0*np.exp(-4*np.log(2)*(abs(VD.ZC[y_inside])/b)**2)

 
    u = u_inviscid - Va_deficit/Vv
    v = v_inviscid
    w = w_inviscid
    
    wing_wake = Data()
    wing_wake.u_velocities = u
    wing_wake.v_velocities = v
    wing_wake.w_velocities = w
    wing_wake.VD           = VD
    
    # Contour plots of the flow field behind the wing
    if plot_wake:
        xplot = grid_points.yline/(0.5*span)
        yplot = grid_points.zline
        zplot_w = np.reshape(w, (len(grid_points.yline),len(grid_points.zline))).T
        zplot_u = np.reshape(u, (len(grid_points.yline),len(grid_points.zline))).T
        zplot_v = np.reshape(v, (len(grid_points.yline),len(grid_points.zline))).T
           
        
        fig  = plt.figure(figsize=(10,4))
        axes = fig.add_subplot(131)
        a = axes.contourf(xplot,yplot,zplot_w, levels=100,cmap='hot')
        axes.set_xlabel('$\dfrac{2y}{b}$')
        axes.set_ylabel('z')
        axes.set_title("Downwash Induced Velocity, w")
        plt.colorbar(a,ax=axes,orientation='horizontal')
        
        axes = fig.add_subplot(132)
        a = axes.contourf(xplot,yplot,zplot_u, levels=100,cmap='gist_heat')
        axes.set_xlabel('$\dfrac{2y}{b}$')
        axes.set_ylabel('z')
        axes.set_title("Streamwise Induced Velocity, u")
        plt.colorbar(a,ax=axes,orientation='horizontal')
        
        axes = fig.add_subplot(133)
        a = axes.contourf(xplot,yplot,zplot_v, levels=100,cmap='gist_heat')
        axes.set_xlabel('$\dfrac{2y}{b}$')
        axes.set_ylabel('z')
        axes.set_title("Spanwise Induced Velocity, v")
        plt.colorbar(a,ax=axes,orientation='horizontal')        
    
    
    return wing_wake