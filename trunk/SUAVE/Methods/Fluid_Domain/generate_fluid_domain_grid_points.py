## @ingroup Methods-Fluid_Domain
# generate_fluid_domain_grid_points.py
#
# Created:  Aug 2022, R. Erhard
# Modified: 
#           

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------
import numpy as np
from SUAVE.Core import Data

def generate_fluid_domain_grid_points(Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, R, dL=0.04, xRbuffer=1., yRbuffer=0.5, zRbuffer=0.5):
    """
    Generates the grid points in cartesian coordinates for a rectangular box with specified dimensions.
    
    Inputs:
       Xmin, Xmax, Ymin, Ymax, Zmin, Zmax    -  Boundary coordinates for the rectangular box
       R                                     -  Characteristic length (radius)
       dL                                    -  Length of each grid cell in fluid domain
       Rbuffer                               -  Additional buffer to expand the domain boundaries
    
    Outputs:
       GridPoints   - Data structure containing the cartesian coordinates of the grid points in the domain
       
    """
    
    Nx = round( (Xmax-Xmin) / dL )
    Ny = round( (Ymax-Ymin) / dL )
    Nz = round( (Zmax-Zmin) / dL )
    
    Xouter = np.linspace(Xmin - R*xRbuffer, Xmax + R*xRbuffer, Nx) 
    Youter = np.linspace(Ymin - R*yRbuffer, Ymax + R*yRbuffer, Ny)
    Zouter = np.linspace(Zmin - R*zRbuffer, Zmax + R*zRbuffer, Nz)
    
    Xp, Yp, Zp = np.meshgrid(Xouter, Youter, Zouter, indexing='ij')
    
    # stack grid points
    Xstacked = np.reshape(Xp, np.size(Xp))
    Ystacked = np.reshape(Yp, np.size(Yp))
    Zstacked = np.reshape(Zp, np.size(Zp))
    
    GridPoints = Data()
    GridPoints.XC = Xstacked
    GridPoints.YC = Ystacked
    GridPoints.ZC = Zstacked
    GridPoints.Xouter = Xouter
    GridPoints.Youter = Youter
    GridPoints.Zouter = Zouter
    GridPoints.Xp = Xp
    GridPoints.Yp = Yp
    GridPoints.Zp = Zp
    GridPoints.n_cp = len(Xstacked)  
    GridPoints.original_shape = np.shape(Xp)
    
    return GridPoints

