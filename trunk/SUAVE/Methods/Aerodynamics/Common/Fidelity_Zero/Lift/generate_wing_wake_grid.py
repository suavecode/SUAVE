## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# generate_wing_wake_grid.py
# 
# Created:   April 2021, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np
import pylab as plt
from SUAVE.Core import Data


def generate_wing_wake_grid(geometry, H, L, hf, x_plane, Nzo=20, Nzf=35, Nyo=20, plot_grid=False):
    """ Generates the grid points for evaluating the viscous wing wake in a downstream plane.
    Uses smaller grid near the wing to better capture boundary layer.
    
    Inputs: 
    geometry     SUAVE vehicle data object
    H            Height of full grid, normalized by wing span
    L            Length of grid, normalized by wing span
    hf           Height of finer grid portion
    x_plane      Spanwise location of grid plane
    
    Nzo          Number of vertical grid points outside finer region
    Nzf          Number of vertical grid points inside finer region
    Nyo          Number of horizontal grid points outside of wing span
    """
    # unpack
    span      = geometry.wings.main_wing.spans.projected
    half_span = span/2
    VD        = geometry.vortex_distribution
    breaks    = VD.chordwise_breaks
    
    # grid bounds
    z_bot     = -H*half_span
    z_top     = H*half_span
    Nzo_half  = int(Nzo/2)
    Nyo_half  = int(Nyo/2)
       
    # generate vertical grid point locations
    z_outer_bot   = np.linspace(z_bot, -hf, Nzo_half)
    z_outer_top   = np.linspace(hf, z_top, Nzo_half)
    
    # use finer concentration of grid points near the wing
    z_inner_bot   = -hf*(np.flipud((1-np.cos(np.linspace(1e-6,1,Nzf)*np.pi/2))))
    z_inner_top   = hf*(1-np.cos(np.linspace(0,1,Nzf)*np.pi/2))
    zlocs         = np.concatenate([z_outer_bot, z_inner_bot, z_inner_top, z_outer_top])

    # generate spanwise grid point locations: placed between vortex lines to avoid discontinuities
    ypts        = VD.YC[breaks]
    y_semispan  = ypts[0:int(len(ypts)/2)]
    
    if L>=1.:
        # add grid points outside wingtip
        y_outerspan = np.linspace(1.01,L,Nyo_half)*half_span
        y_semispan  = np.append(y_semispan, y_outerspan)
    else:
        # trim spanwise points to region of interest
        y_in       = y_semispan<(L*half_span)
        y_semispan = y_semispan[y_in]
        
    ylocs       = np.concatenate([np.flipud(-y_semispan),y_semispan])
    
    # declare new control points
    cp_YC = np.repeat(ylocs,len(zlocs)) 
    cp_ZC = np.tile(zlocs,len(ylocs))
    cp_XC = np.ones_like(cp_YC)*x_plane 
    
    grid_points = Data()
    grid_points.XC = cp_XC
    grid_points.YC = cp_YC
    grid_points.ZC = cp_ZC
    grid_points.yline = ylocs
    grid_points.zline = zlocs
    
    if plot_grid:
        yL = -span/2
        yR = span/2
        
        wing_y = np.array([yL, yR])
        wing_z = np.array([0,0])
        
        # plot the grid points
        fig  = plt.figure()
        axes = fig.add_subplot(1,1,1)
        axes.plot(cp_YC,cp_ZC,'k.')
        
        # plot the wing projection
        axes.plot(wing_y,wing_z, 'r')
        
        axes.set_xlabel('y [m]')
        axes.set_ylabel("z [m]")
        axes.set_title("New Grid Points")
        
        plot_prop=True
        if plot_prop:
            for net in list(geometry.networks.keys()):
                for prop in list(geometry.networks[net].propellers.keys()):
                    R      = geometry.networks[net].propellers[prop].tip_radius
                    origin = geometry.networks[net].propellers[prop].origin
                    Na     = geometry.networks[net].propellers[prop].number_azimuthal_stations
                    
                    psi    = np.linspace(0,2*np.pi,Na+1)[:-1]
                    ycoords = origin[0][1] + R*np.cos(psi)
                    zcoords = origin[0][2] + R*np.sin(psi)
                    axes.plot(ycoords,zcoords,'r')
        
    return grid_points