## @ingroup Plots-Geometry-Three_Dimensional 
# plot_3d_nacelle.py
#
# Created : Mar 2020, M. Clarke
# Modified: Apr 2020, M. Clarke
# Modified: Jul 2020, M. Clarke
# Modified: Jul 2021, E. Botero
# Modified: Oct 2021, M. Clarke
# Modified: Dec 2021, M. Clarke
# Modified: Feb 2022, R. Erhard
# Modified: Mar 2022, R. Erhard
# Modified: Sep 2022, M. Clarke
# Modified: Nov 2022, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
import numpy as np 
import plotly.graph_objects as go 
from SUAVE.Plots.Geometry.Common.contour_surface_slice import contour_surface_slice
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry import import_airfoil_geometry
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_naca_4series    import compute_naca_4series 

## @ingroup Plots-Geometry-Three_Dimensional 
def plot_3d_nacelle(plot_data,nacelle,tessellation = 24,number_of_airfoil_points = 21,color_map= 'darkmint'):
    """ This plots a 3D surface of a nacelle  

    Assumptions:
    None

    Source:
    None 
    
    Inputs:
    axes                       - plotting axes 
    nacelle                    - SUAVE nacelle data structure
    tessellation               - azimuthal discretization of lofted body 
    number_of_airfoil_points   - discretization of airfoil geometry 
    color_map                  - face color of nacelle  

    Properties Used:
    N/A
    """

    nac_pts = generate_3d_nacelle_points(nacelle,tessellation,number_of_airfoil_points)    

    num_nac_segs = len(nac_pts[:,0,0])
    tesselation  = len(nac_pts[0,:,0]) 
    for i_seg in range(num_nac_segs-1):
        for i_tes in range(tesselation-1):
            X = np.array([[nac_pts[i_seg  ,i_tes  ,0],nac_pts[i_seg+1,i_tes  ,0]],
                 [nac_pts[i_seg  ,i_tes+1,0],nac_pts[i_seg+1,i_tes+1,0]]])
            Y = np.array([[nac_pts[i_seg  ,i_tes  ,1],nac_pts[i_seg+1,i_tes  ,1]],
                 [nac_pts[i_seg  ,i_tes+1,1],nac_pts[i_seg+1,i_tes+1,1]]])
            Z = np.array([[nac_pts[i_seg  ,i_tes  ,2],nac_pts[i_seg+1,i_tes  ,2]],
                 [nac_pts[i_seg  ,i_tes+1,2],nac_pts[i_seg+1,i_tes+1,2]]])
             
            values = np.zeros_like(X) 
            verts = contour_surface_slice(X, Y, Z ,values,color_map)
            plot_data.append(verts)    

    return plot_data

## @ingroup Plots-Geometry-Three_Dimensional 
def generate_3d_nacelle_points(nac,tessellation = 24 ,number_of_airfoil_points = 21):
    """ This generates the coordinate points on the surface of the nacelle

    Assumptions:
    None

    Source:
    None

    Inputs:
    nac                        - Nacelle data structure 
    tessellation               - azimuthal discretization of lofted body 
    number_of_airfoil_points   - discretization of airfoil geometry 
    
    Properties Used:
    N/A 
    """ 
    
    num_nac_segs = len(nac.Segments.keys())   
    theta        = np.linspace(0,2*np.pi,tessellation) 
    
    if num_nac_segs == 0:
        num_nac_segs = int(np.ceil(number_of_airfoil_points/2))
        nac_pts      = np.zeros((num_nac_segs,tessellation,3))
        naf          = nac.Airfoil
        
        if naf.NACA_4_series_flag == True:  
            a_geo        = compute_naca_4series(naf.coordinate_file,num_nac_segs)
            xpts         = np.repeat(np.atleast_2d(a_geo.x_coordinates[0]).T,tessellation,axis = 1)*nac.length
            zpts         = np.repeat(np.atleast_2d(a_geo.y_coordinates[0]).T,tessellation,axis = 1)*nac.length  
        
        elif naf.coordinate_file != None: 
            a_geo        = import_airfoil_geometry(naf.coordinate_file,num_nac_segs)
            xpts         = np.repeat(np.atleast_2d(np.take(a_geo.x_coordinates,[0],axis=0)).T,tessellation,axis = 1)*nac.length
            zpts         = np.repeat(np.atleast_2d(np.take(a_geo.y_coordinates,[0],axis=0)).T,tessellation,axis = 1)*nac.length
        
        else:
            # if no airfoil defined, use super ellipse as default
            a =  nac.length/2 
            b =  (nac.diameter - nac.inlet_diameter)/2 
            b = np.maximum(b,1E-3) # ensure 
            xpts =  np.repeat(np.atleast_2d(np.linspace(-a,a,num_nac_segs)).T,tessellation,axis = 1) 
            zpts = (np.sqrt((b**2)*(1 - (xpts**2)/(a**2) )))*nac.length 
            xpts = (xpts+a)*nac.length  

        if nac.flow_through: 
            zpts = zpts + nac.inlet_diameter/2  
                
        # create geometry 
        theta_2d = np.repeat(np.atleast_2d(theta),num_nac_segs,axis =0) 
        nac_pts[:,:,0] =  xpts
        nac_pts[:,:,1] =  zpts*np.cos(theta_2d)
        nac_pts[:,:,2] =  zpts*np.sin(theta_2d)  
                
    else:
        nac_pts = np.zeros((num_nac_segs,tessellation,3)) 
        for i_seg in range(num_nac_segs):
            a        = nac.Segments[i_seg].width/2
            b        = nac.Segments[i_seg].height/2
            r        = np.sqrt((b*np.sin(theta))**2  + (a*np.cos(theta))**2)
            nac_ypts = r*np.cos(theta)
            nac_zpts = r*np.sin(theta)
            nac_pts[i_seg,:,0] = nac.Segments[i_seg].percent_x_location*nac.length
            nac_pts[i_seg,:,1] = nac_ypts + nac.Segments[i_seg].percent_y_location*nac.length 
            nac_pts[i_seg,:,2] = nac_zpts + nac.Segments[i_seg].percent_z_location*nac.length  
            
    # rotation about y to orient propeller/rotor to thrust angle
    rot_trans =  nac.nac_vel_to_body()
    rot_trans =  np.repeat( np.repeat(rot_trans[ np.newaxis,:,: ],tessellation,axis=0)[ np.newaxis,:,:,: ],num_nac_segs,axis=0)    
    
    NAC_PTS  =  np.matmul(rot_trans,nac_pts[...,None]).squeeze()  
     
    # translate to body 
    NAC_PTS[:,:,0] = NAC_PTS[:,:,0] + nac.origin[0][0]
    NAC_PTS[:,:,1] = NAC_PTS[:,:,1] + nac.origin[0][1]
    NAC_PTS[:,:,2] = NAC_PTS[:,:,2] + nac.origin[0][2]
     
    return NAC_PTS 