## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# generate_propeller_grid.py
# 
# Created:   April 2021, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np
import pylab as plt
from SUAVE.Core import Data

def generate_propeller_grid(prop, grid_settings, plot_grid=True):
    """ Generates new grid points at a propeller plane.
    
    Inputs: 
    prop                     SUAVE vehicle data object
    grid_settings            Settings for the grid
         radius              Radius for cartesian grid (propeller radius)
         hub_radius          Radius for non-evaluation points (propeller hub radius)
         grid_mode           Option for cartesian or radial grid description
         Ny                  Number of points in y-direction for cartesian grid
         Nz                  Number of points in z-direction for cartesian grid
         
    plot_grid                Boolean for visualizing the grid point generation

    
    grid_points              Dictionary of grid points for the propeller grid

    """    
    R         = grid_settings.radius
    Rh        = grid_settings.hub_radius
    Nr        = grid_settings.Nr
    Na        = grid_settings.Na
    grid_mode = grid_settings.grid_mode
    Ny        = grid_settings.Ny
    Nz        = grid_settings.Nz
    psi_360   = np.linspace(0,2*np.pi,Na+1)
    influencing_prop = prop.origin[0]
    influenced_prop  = prop.origin[1]
    
    y_offset         = influenced_prop[1] - influencing_prop[1] 
    z_offset         = influenced_prop[2] - influencing_prop[2] 

    
    if grid_mode == 'radial':
        psi     = psi_360[:-1]
        psi_2d  = np.tile(np.atleast_2d(psi).T,(1,Nr)) 
        r       = np.linspace(Rh,0.99*R,Nr)
        
        # basic radial grid
        ymesh = r*np.cos(psi_2d)
        zmesh = r*np.sin(psi_2d)
        
    elif grid_mode == 'cartesian':
        y     = np.linspace(-R,R,Ny)
        z     = np.linspace(-R,R,Nz)
        ymesh = np.tile(np.atleast_2d(y).T,(1,Nz))
        zmesh = np.tile(np.atleast_2d(z),(Ny,1))
        
        r_pts   = np.sqrt(ymesh**2 + zmesh**2)
        cutoffs = r_pts<R
        
        ymesh  = ymesh[cutoffs]
        zmesh  = zmesh[cutoffs]
        
    
    grid_points        = Data()
    grid_points.ymesh  = ymesh + y_offset
    grid_points.zmesh  = zmesh + z_offset
    grid_points.Nr     = Nr
    grid_points.Na     = Na
    
    if plot_grid:
        
        # plot the grid points
        fig  = plt.figure()
        axes = fig.add_subplot(1,1,1)
        axes.plot(ymesh,zmesh,'k.')
        
        # plot the propeller radius
        axes.plot(R*np.cos(psi_360), R*np.sin(psi_360), 'r')
        
        axes.set_aspect('equal', 'box')
        axes.set_xlabel('y [m]')
        axes.set_ylabel("z [m]")
        axes.set_title("New Grid Points")
    
    return grid_points