## @ingroup Visualization-Geometry-Three_Dimensional
#  plot_3d_rotor.py
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
from MARC.Core import Data
import numpy as np 
import plotly.graph_objects as go  
from plotly.express.colors import sample_colorscale   
from MARC.Visualization.Geometry.Common.contour_surface_slice import contour_surface_slice
from MARC.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry import import_airfoil_geometry
from MARC.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_naca_4series    import compute_naca_4series


## @ingroup Visualization-Geometry-Three_Dimensional  
def plot_3d_rotor(rotor,save_filename = "Rotor", save_figure = False, plot_data = None,plot_axis = False, cpt=0, number_of_airfoil_points = 21,
                            color_map='turbid',alpha=1):
    """ This plots a 3D surface of the  rotor

    Assumptions:
    None

    Source:
    None

    Inputs:
    axes                       - plotting axes
    rotor                      - MARC rotor for which to plot the geometry
    cpt                        - control point at which to plot the rotor
    number_of_airfoil_points   - discretization of airfoil geometry 
    

    Properties Used:
    N/A
    """
    plot_propeller_only = False
    if plot_data == None: 
        print("\nPlotting rotor") 
    
        plot_propeller_only = True         
        camera        = dict(up=dict(x=0.5, y=0.5, z=1), center=dict(x=0, y=0, z=-0.5), eye=dict(x=1.5, y=1.5, z=.8))
        plot_data     = []
        
    num_B     = rotor.number_of_blades 
    af_pts    = number_of_airfoil_points-1
    dim       = len(rotor.radius_distribution)

    for i in range(num_B):
        G = get_3d_blade_coordinates(rotor,number_of_airfoil_points,dim,i)
        # ------------------------------------------------------------------------
        # Plot Rotor Blade
        # ------------------------------------------------------------------------
        for sec in range(dim-1):
            for loc in range(af_pts):
                X = np.array([[G.XA1[cpt,sec,loc],G.XA2[cpt,sec,loc]],
                     [G.XB1[cpt,sec,loc],G.XB2[cpt,sec,loc]]])
                Y = np.array([[G.YA1[cpt,sec,loc],G.YA2[cpt,sec,loc]],
                     [G.YB1[cpt,sec,loc],G.YB2[cpt,sec,loc]]])
                Z = np.array([[G.ZA1[cpt,sec,loc],G.ZA2[cpt,sec,loc]],
                     [G.ZB1[cpt,sec,loc],G.ZB2[cpt,sec,loc]]]) 
                 
                values      = np.ones_like(X) 
                verts       = contour_surface_slice(X, Y, Z ,values,color_map)
                plot_data.append(verts)      
            
    axis_limits = np.maximum(np.max(G.XA1), np.maximum(np.max(G.YA1),np.max(G.ZA1)))*2 
    if plot_propeller_only:
        fig = go.Figure(data=plot_data)
        fig.update_scenes(aspectmode   = 'auto',
                          xaxis_visible=plot_axis,
                          yaxis_visible=plot_axis,
                          zaxis_visible=plot_axis
                          )
        fig.update_layout( 
                 width     = 1500,
                 height    = 1500, 
                 scene = dict(
                            xaxis = dict(backgroundcolor="lightgrey", gridcolor="white", showbackground=plot_axis,
                                         zerolinecolor="white", range=[-axis_limits,axis_limits]),
                            yaxis = dict(backgroundcolor="lightgrey", gridcolor="white", showbackground=plot_axis, 
                                         zerolinecolor="white", range=[-axis_limits,axis_limits]),
                            zaxis = dict(backgroundcolor="lightgrey",gridcolor="white",showbackground=plot_axis,
                                         zerolinecolor="white", range=[-axis_limits,axis_limits])),             
                 scene_camera=camera) 
        fig.update_coloraxes(showscale=False)
        fig.update_traces(opacity = alpha) 
        if save_figure: 
            fig.write_image(save_filename + ".png")
        fig.show()
        return 
    else: 
        return plot_data

## @ingroup Visualization-Geometry-Three_Dimensional 
def get_3d_blade_coordinates(rotor,n_points,dim,i,aircraftRefFrame=True):
    """ This generates the coordinates of the blade surface for plotting in the aircraft frame (x-back, z-up)

    Assumptions:
    None

    Source:
    None

    Inputs:
    rotor            - MARC rotor
    n_points         - number of points around airfoils of each blade section
    dim              - number for radial dimension
    i                - blade number
    aircraftRefFrame - boolean to convert the coordinates from rotor frame to aircraft frame 

    Properties Used:
    N/A
    """    
    # unpack 
    num_B        = rotor.number_of_blades
    airfoils     = rotor.Airfoils 
    beta         = rotor.twist_distribution + rotor.inputs.pitch_command  
    a_o          = rotor.start_angle
    b            = rotor.chord_distribution
    r            = rotor.radius_distribution
    MCA          = rotor.mid_chord_alignment
    t            = rotor.max_thickness_distribution
    a_loc        = rotor.airfoil_polar_stations
    origin       = rotor.origin

    if rotor.rotation==1:
        # negative chord and twist to give opposite rotation direction
        b    = -b    
        beta = -beta

    theta  = np.linspace(0,2*np.pi,num_B+1)[:-1] 
    flip_2 =  (np.pi/2)

    MCA_2d             = np.repeat(np.atleast_2d(MCA).T,n_points,axis=1)
    b_2d               = np.repeat(np.atleast_2d(b).T  ,n_points,axis=1)
    t_2d               = np.repeat(np.atleast_2d(t).T  ,n_points,axis=1)
    r_2d               = np.repeat(np.atleast_2d(r).T  ,n_points,axis=1)
    airfoil_le_offset  = np.repeat(b[:,None], n_points, axis=1)/2  

    # get airfoil coordinate geometry
    if len(airfoils.keys())>0:
        xpts  = np.zeros((dim,n_points))
        zpts  = np.zeros((dim,n_points))
        max_t = np.zeros(dim)
        for af_idx,airfoil in enumerate(airfoils):
            geometry     = import_airfoil_geometry(airfoil.coordinate_file,n_points)
            locs         = np.where(np.array(a_loc) == af_idx)
            xpts[locs]   = geometry.x_coordinates  
            zpts[locs]   = geometry.y_coordinates  
            max_t[locs]  = geometry.thickness_to_chord 

    else: 
        airfoil_data = compute_naca_4series('2410',n_points)
        xpts         = np.repeat(np.atleast_2d(airfoil_data.x_coordinates) ,dim,axis=0)
        zpts         = np.repeat(np.atleast_2d(airfoil_data.y_coordinates) ,dim,axis=0)
        max_t        = np.repeat(airfoil_data.thickness_to_chord,dim,axis=0)

    # store points of airfoil in similar format as Vortex Points (i.e. in vertices)
    max_t2d = np.repeat(np.atleast_2d(max_t).T ,n_points,axis=1)

    xp      = (- MCA_2d + xpts*b_2d - airfoil_le_offset)     # x-coord of airfoil
    yp      = r_2d*np.ones_like(xp)                          # radial location
    zp      = zpts*(t_2d/max_t2d)                            # former airfoil y coord

    rotor_vel_to_body = rotor.prop_vel_to_body()
    cpts              = len(rotor_vel_to_body[:,0,0])

    matrix        = np.zeros((len(zp),n_points,3)) # radial location, airfoil pts (same y)
    matrix[:,:,0] = xp
    matrix[:,:,1] = yp
    matrix[:,:,2] = zp
    matrix        = np.repeat(matrix[None,:,:,:], cpts, axis=0)


    # ROTATION MATRICES FOR INNER SECTION
    # rotation about y axis to create twist and position blade upright
    trans_1        = np.zeros((dim,3,3))
    trans_1[:,0,0] = np.cos(- beta)
    trans_1[:,0,2] = -np.sin(- beta)
    trans_1[:,1,1] = 1
    trans_1[:,2,0] = np.sin(- beta)
    trans_1[:,2,2] = np.cos(- beta)
    trans_1        = np.repeat(trans_1[None,:,:,:], cpts, axis=0)

    # rotation about x axis to create azimuth locations
    trans_2 = np.array([[1 , 0 , 0],
                        [0 , np.cos(theta[i] + a_o + flip_2 ), -np.sin(theta[i] +a_o +  flip_2)],
                        [0,np.sin(theta[i] + a_o + flip_2), np.cos(theta[i] + a_o + flip_2)]])
    trans_2 = np.repeat(trans_2[None,:,:], dim, axis=0)
    trans_2 = np.repeat(trans_2[None,:,:,:], cpts, axis=0)

    # rotation about y to orient propeller/rotor to thrust angle (from propeller frame to aircraft frame)
    trans_3 =  rotor_vel_to_body
    trans_3 =  np.repeat(trans_3[:, None,:,: ],dim,axis=1) 

    trans     = np.matmul(trans_2,trans_1)
    rot_mat   = np.repeat(trans[:,:, None,:,:],n_points,axis=2)    

    # ---------------------------------------------------------------------------------------------
    # ROTATE POINTS
    if aircraftRefFrame:
        # rotate all points to the thrust angle with trans_3
        mat  =  np.matmul(np.matmul(rot_mat,matrix[...,None]).squeeze(axis=-1), trans_3)
    else:
        # use the rotor frame
        mat  =  np.matmul(rot_mat,matrix[...,None]).squeeze(axis=-1)
    # ---------------------------------------------------------------------------------------------
    # create empty data structure for storing geometry
    G = Data()

    # store node points
    G.X  = mat[:,:,:,0] + origin[0][0]
    G.Y  = mat[:,:,:,1] + origin[0][1]
    G.Z  = mat[:,:,:,2] + origin[0][2]

    # store points
    G.XA1  = mat[:,:-1,:-1,0] + origin[0][0]
    G.YA1  = mat[:,:-1,:-1,1] + origin[0][1]
    G.ZA1  = mat[:,:-1,:-1,2] + origin[0][2]
    G.XA2  = mat[:,:-1,1:,0]  + origin[0][0]
    G.YA2  = mat[:,:-1,1:,1]  + origin[0][1]
    G.ZA2  = mat[:,:-1,1:,2]  + origin[0][2]

    G.XB1  = mat[:,1:,:-1,0] + origin[0][0]
    G.YB1  = mat[:,1:,:-1,1] + origin[0][1]
    G.ZB1  = mat[:,1:,:-1,2] + origin[0][2]
    G.XB2  = mat[:,1:,1:,0]  + origin[0][0]
    G.YB2  = mat[:,1:,1:,1]  + origin[0][1]
    G.ZB2  = mat[:,1:,1:,2]  + origin[0][2]    
    
    return G
 