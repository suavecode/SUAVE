## @ingroup Plots-Geometry
# plot_vehicle.py
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
from SUAVE.Core import Data
import numpy as np 
import plotly.graph_objects as go 
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry import import_airfoil_geometry
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_naca_4series    import compute_naca_4series
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.generate_vortex_distribution    import generate_vortex_distribution 
from SUAVE.Analyses.Aerodynamics import Vortex_Lattice

## @ingroup Plots-Geometry
def plot_vehicle(vehicle,plot_axis = False, save_figure = False, plot_control_points = True, save_filename = "Vehicle_Geometry"):
    """This plots vortex lattice panels created when Fidelity Zero  Aerodynamics
    Routine is initialized

    Assumptions:
    None

    Source:
    None

    Inputs:
    vehicle

    Outputs:
    Plots

    Properties Used:
    N/A
    """

    print("\nPlotting vehicle") 
    camera        = dict(up=dict(x=0.5, y=0.5, z=1), center=dict(x=0, y=0, z=-0.5), eye=dict(x=-1.5, y=-1.5, z=.8))
    plot_data     = []
    
    # unpack vortex distribution
    try:
        VD = vehicle.vortex_distribution
    except:
        settings = Vortex_Lattice().settings
        settings.number_spanwise_vortices  = 25
        settings.number_chordwise_vortices = 5
        settings.spanwise_cosine_spacing   = False
        settings.model_fuselage            = False
        settings.model_nacelle             = False
        VD = generate_vortex_distribution(vehicle,settings)


    # -------------------------------------------------------------------------
    # PLOT WING
    # -------------------------------------------------------------------------
    wing_face_color = 'darkgrey'
    wing_edge_color = 'grey'
    wing_alpha_val  = 1
    plot_data       = plot_wing(plot_data,VD,wing_face_color,wing_edge_color,wing_alpha_val)
    if  plot_control_points: 
        ctrl_pts = go.Scatter3d(x=VD.XC, y=VD.YC, z=VD.ZC,
                                    mode  = 'markers',
                                    marker= dict(size=6,color='red',opacity=0.8),
                                    line  = dict(color='red',width=2))
        plot_data.append(ctrl_pts)
        

    # -------------------------------------------------------------------------
    # PLOT WAKES
    # -------------------------------------------------------------------------
    wake_face_color = 'white'
    wake_edge_color = 'blue'
    wake_alpha      = 0.5
    for net in vehicle.networks:
        if "propellers" in net.keys():
            for prop in net.propellers:
                # plot propeller wake
                if prop.Wake.wake_method =="Fidelity_One":
                    plot_data = plot_propeller_wake(plot_data, prop, wake_face_color, wake_edge_color, wake_alpha)
        if "lift_rotors" in net.keys():
            for rot in net.lift_rotors:
                # plot rotor wake
                if rot.Wake.wake_method =="Fidelity_One":
                    plot_data = plot_propeller_wake(plot_data, rot, wake_face_color, wake_edge_color, wake_alpha)            
            

    # -------------------------------------------------------------------------
    # PLOT FUSELAGE
    # -------------------------------------------------------------------------
    fuselage_face_color = 'grey'
    fuselage_edge_color = 'darkgrey'
    fuselage_alpha      = 1
    for fus in vehicle.fuselages:
        # Generate Fuselage Geometry
        fus_pts = generate_fuselage_points(fus)

        # Plot Fuselage Geometry
        plot_data = plot_fuselage_geometry(plot_data,fus_pts,fuselage_face_color,fuselage_edge_color,fuselage_alpha)

    # -------------------------------------------------------------------------
    # PLOT ENGINE
    # -------------------------------------------------------------------------
    nacelle_face_color       = 'darkred'
    nacelle_edge_color       = 'black'
    nacelle_alpha            = 1
    number_of_airfoil_points = 21
    for nacelle in vehicle.nacelles:  
        # Generate Nacelle Geoemtry
        nac_geo = generate_nacelle_points(nacelle,number_of_airfoil_points)
        
        # Plot Nacelle Geometry
        plot_data = plot_nacelle_geometry(plot_data,nac_geo,nacelle_face_color,nacelle_edge_color,nacelle_alpha ) 

           
    # -------------------------------------------------------------------------
    # PLOT ENGINE
    # -------------------------------------------------------------------------
    network_face_color       = 'darkred'
    network_edge_color       = 'red'
    network_alpha            = 1
    number_of_airfoil_points = 21
    for network in vehicle.networks:
        plot_data = plot_network(plot_data,network,number_of_airfoil_points,network_face_color,network_edge_color,network_alpha )

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
                        xaxis = dict(backgroundcolor="grey", gridcolor="white", showbackground=plot_axis, zerolinecolor="white",),
                        yaxis = dict(backgroundcolor="grey", gridcolor="white", showbackground=plot_axis, zerolinecolor="white"),
                        zaxis = dict(backgroundcolor="grey",gridcolor="white",showbackground=plot_axis,zerolinecolor="white",)),             
             scene_camera=camera) 
    fig.update_coloraxes(showscale=False)
    
    if save_figure:
        fig.write_image(save_filename + ".png")
    fig.show()
        
    return

def plot_wing(plot_data,VD,face_color,edge_color,alpha_val):
    """ This plots the wings of a vehicle

    Assumptions:
    None

    Source:
    None

    Inputs:
    VD.
       XA1...ZB2         - coordinates of wing vortex distribution
    face_color           - color of panel
    edge_color           - color of panel edge
    alpha_color          - translucency:  1 = opaque , 0 = transparent

    Properties Used:
    N/A
    """

    n_cp = VD.n_cp
    for i in range(n_cp): 

        X = np.array([[VD.XA1[i],VD.XA2[i]],[VD.XB1[i],VD.XB2[i]]])
        Y = np.array([[VD.YA1[i],VD.YA2[i]],[VD.YB1[i],VD.YB2[i]]])
        Z = np.array([[VD.ZA1[i],VD.ZA2[i]],[VD.ZB1[i],VD.ZB2[i]]])           
        
        values      = np.ones_like(X)*0.5
        color_scale = 'greys'
        verts       = contour_surface_slice(X, Y, Z ,values,color_scale)
        plot_data.append(verts)         

    return plot_data

def plot_propeller_wake(plot_data, prop,face_color,edge_color,alpha,ctrl_pt=0):
    """ This plots a helical wake of a propeller or rotor

    Assumptions:
    None

    Source:
    None

    Inputs:
    VD.Wake.
       XA1...ZB2         - coordinates of wake vortex distribution
    face_color           - color of panel
    edge_color           - color of panel edge
    alpha_color          - translucency:  1 = opaque , 0 = transparent

    Properties Used:
    N/A
    """
    wVD      = prop.Wake.vortex_distribution.reshaped_wake 
    num_B    = len(wVD.XA1[0,0,:,0,0])
    dim_R    = len(wVD.XA1[0,0,0,:,0])
    nts      = len(wVD.XA1[0,0,0,0,:])
    
    for t_idx in range(nts):
        for B_idx in range(num_B):
            for loc in range(dim_R):
                X = np.array([[wVD.XA1[0,ctrl_pt,B_idx,loc,t_idx],wVD.XA2[0,ctrl_pt,B_idx,loc,t_idx]],
                              [ wVD.XB1[0,ctrl_pt,B_idx,loc,t_idx],wVD.XB2[0,ctrl_pt,B_idx,loc,t_idx]]])
                Y = np.array([[wVD.YA1[0,ctrl_pt,B_idx,loc,t_idx], wVD.YA2[0,ctrl_pt,B_idx,loc,t_idx]],
                              [ wVD.YB1[0,ctrl_pt,B_idx,loc,t_idx],wVD.YB2[0,ctrl_pt,B_idx,loc,t_idx]]])
                Z = np.array([[wVD.ZA1[0,ctrl_pt,B_idx,loc,t_idx],wVD.ZA2[0,ctrl_pt,B_idx,loc,t_idx]],
                              [wVD.ZB1[0,ctrl_pt,B_idx,loc,t_idx],wVD.ZB2[0,ctrl_pt,B_idx,loc,t_idx]]])
        
                values = np.ones_like(X)
                color_scale = 'Blues'
                verts = contour_surface_slice(X, Y, Z ,values,color_scale)
                plot_data.append(verts)     
                 
    return plot_data


def generate_fuselage_points(fus ,tessellation = 24 ):
    """ This generates the coordinate points on the surface of the fuselage

    Assumptions:
    None

    Source:
    None

    Inputs:
    fus                  - fuselage data structure

    Properties Used:
    N/A
    """
    num_fus_segs = len(fus.Segments.keys())
    fus_pts      = np.zeros((num_fus_segs,tessellation ,3))

    if num_fus_segs > 0:
        for i_seg in range(num_fus_segs):
            theta    = np.linspace(0,2*np.pi,tessellation)
            a        = fus.Segments[i_seg].width/2
            b        = fus.Segments[i_seg].height/2
            r        = np.sqrt((b*np.sin(theta))**2  + (a*np.cos(theta))**2)
            fus_ypts = r*np.cos(theta)
            fus_zpts = r*np.sin(theta)
            fus_pts[i_seg,:,0] = fus.Segments[i_seg].percent_x_location*fus.lengths.total + fus.origin[0][0]
            fus_pts[i_seg,:,1] = fus_ypts + fus.Segments[i_seg].percent_y_location*fus.lengths.total + fus.origin[0][1]
            fus_pts[i_seg,:,2] = fus_zpts + fus.Segments[i_seg].percent_z_location*fus.lengths.total + fus.origin[0][2]

    return fus_pts


def plot_fuselage_geometry(plot_data,fus_pts, face_color,edge_color,alpha):
    """ This plots the 3D surface of the fuselage

    Assumptions:
    None

    Source:
    None

    Inputs:
    fus_pts              - coordinates of fuselage points
    face_color           - color of panel
    edge_color           - color of panel edge
    alpha_color          - translucency:  1 = opaque , 0 = transparent


    Properties Used:
    N/A
    """

    num_fus_segs = len(fus_pts[:,0,0])
    if num_fus_segs > 0:
        tesselation  = len(fus_pts[0,:,0])
        for i_seg in range(num_fus_segs-1):
            for i_tes in range(tesselation-1):
                X = np.array([[fus_pts[i_seg  ,i_tes,0],fus_pts[i_seg+1,i_tes  ,0]],
                              [fus_pts[i_seg  ,i_tes+1,0],fus_pts[i_seg+1,i_tes+1,0]]])
                Y = np.array([[fus_pts[i_seg  ,i_tes  ,1],fus_pts[i_seg+1,i_tes  ,1]],
                              [fus_pts[i_seg  ,i_tes+1,1],fus_pts[i_seg+1,i_tes+1,1]]])
                Z = np.array([[fus_pts[i_seg  ,i_tes  ,2],fus_pts[i_seg+1,i_tes  ,2]],
                              [fus_pts[i_seg  ,i_tes+1,2],fus_pts[i_seg+1,i_tes+1,2]]])  
                values = np.ones_like(X)*0.5
                color_scale = 'greys'
                verts = contour_surface_slice(X, Y, Z ,values,color_scale)
                plot_data.append(verts)          

    return plot_data


def plot_network(plot_data,network,number_of_airfoil_points,prop_face_color,prop_edge_color,prop_alpha):
    """ This plots the 3D surface of the network

    Assumptions:
    None

    Source:
    None

    Inputs:
    network            - network data structure
    network_face_color - color of panel
    network_edge_color - color of panel edge
    network_alpha      - translucency:  1 = opaque , 0 = transparent

    Properties Used:
    N/A
    """ 
    plot_axis     = False 
    save_figure   = False 
    save_filename = 'Rotor'
    if ('propellers' in network.keys()):

        for prop in network.propellers:

            # Generate And Plot Propeller/Rotor Geometry 
            plot_data = plot_rotor_geometry(prop,save_filename,save_figure,plot_data,plot_axis,0,number_of_airfoil_points,prop_face_color,prop_edge_color,prop_alpha)

    if ('lift_rotors' in network.keys()):

        for rotor in network.lift_rotors:

            # Generate and Plot Propeller/Rotor Geometry
            plot_data = plot_rotor_geometry(rotor,save_filename,save_figure,plot_data,plot_axis,0,number_of_airfoil_points,prop_face_color,prop_edge_color,prop_alpha)

    return plot_data

def generate_nacelle_points(nac,tessellation = 24,number_of_airfoil_points = 21):
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

def plot_nacelle_geometry(plot_data,NAC_SURF_PTS,face_color,edge_color,alpha):
    """ This plots a 3D surface of a nacelle  

    Assumptions:
    None

    Source:
    None 
    
    Inputs:
    axes          - plotting axes
    NAC_SURF_PTS  - nacelle surface points 
    face_color    - face color of nacelle 
    edge_color    - edge color of nacelle 
    alpha         - transparency factor 

    Properties Used:
    N/A
    """

    num_nac_segs = len(NAC_SURF_PTS[:,0,0])
    tesselation  = len(NAC_SURF_PTS[0,:,0]) 
    for i_seg in range(num_nac_segs-1):
        for i_tes in range(tesselation-1):
            X = np.array([[NAC_SURF_PTS[i_seg  ,i_tes  ,0],NAC_SURF_PTS[i_seg+1,i_tes  ,0]],
                 [NAC_SURF_PTS[i_seg  ,i_tes+1,0],NAC_SURF_PTS[i_seg+1,i_tes+1,0]]])
            Y = np.array([[NAC_SURF_PTS[i_seg  ,i_tes  ,1],NAC_SURF_PTS[i_seg+1,i_tes  ,1]],
                 [NAC_SURF_PTS[i_seg  ,i_tes+1,1],NAC_SURF_PTS[i_seg+1,i_tes+1,1]]])
            Z = np.array([[NAC_SURF_PTS[i_seg  ,i_tes  ,2],NAC_SURF_PTS[i_seg+1,i_tes  ,2]],
                 [NAC_SURF_PTS[i_seg  ,i_tes+1,2],NAC_SURF_PTS[i_seg+1,i_tes+1,2]]])
             
            values = np.ones_like(X)
            color_scale = 'Reds'
            verts = contour_surface_slice(X, Y, Z ,values,color_scale)
            plot_data.append(verts)    

    return plot_data

def plot_rotor_geometry(prop,save_filename = "Rotor", save_figure = False, plot_data = None,plot_axis = False, cpt=0,number_of_airfoil_points = 21,
                            prop_face_color='red',prop_edge_color='darkred',prop_alpha=1):
    """ This plots a 3D surface of the  propeller

    Assumptions:
    None

    Source:
    None

    Inputs:
    axes                       - plotting axes
    prop                       - SUAVE propeller for which to plot the geometry
    cpt                        - control point at which to plot the propeller
    number_of_airfoil_points   - discretization of airfoil geometry 
    

    Properties Used:
    N/A
    """
    plot_propeller_only = False
    if plot_data == None: 
        print("\nPlotting propeller") 
    
        plot_propeller_only = True         
        camera        = dict(up=dict(x=0.5, y=0.5, z=1), center=dict(x=0, y=0, z=-0.5), eye=dict(x=-1.5, y=-1.5, z=.8))
        plot_data     = []
        
    num_B     = prop.number_of_blades 
    af_pts    = number_of_airfoil_points-1
    dim       = len(prop.radius_distribution)

    for i in range(num_B):
        G = get_blade_coordinates(prop,number_of_airfoil_points,dim,i)
        # ------------------------------------------------------------------------
        # Plot Propeller Blade
        # ------------------------------------------------------------------------
        for sec in range(dim-1):
            for loc in range(af_pts):
                X = np.array([[G.XA1[cpt,sec,loc],G.XA2[cpt,sec,loc]],
                     [G.XB1[cpt,sec,loc],G.XB2[cpt,sec,loc]]])
                Y = np.array([[G.YA1[cpt,sec,loc],G.YA2[cpt,sec,loc]],
                     [G.YB1[cpt,sec,loc],G.YB2[cpt,sec,loc]]])
                Z = np.array([[G.ZA1[cpt,sec,loc],G.ZA2[cpt,sec,loc]],
                     [G.ZB1[cpt,sec,loc],G.ZB2[cpt,sec,loc]]]) 
                values = np.ones_like(X)
                color_scale = 'Reds'
                verts = contour_surface_slice(X, Y, Z ,values,color_scale)
                plot_data.append(verts)      
                
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
                            xaxis = dict(backgroundcolor="grey", gridcolor="white", showbackground=plot_axis, zerolinecolor="white",),
                            yaxis = dict(backgroundcolor="grey", gridcolor="white", showbackground=plot_axis, zerolinecolor="white"),
                            zaxis = dict(backgroundcolor="grey",gridcolor="white",showbackground=plot_axis,zerolinecolor="white",)),             
                 scene_camera=camera) 
        fig.update_coloraxes(showscale=False)
        
        if save_figure: 
            fig.write_image(save_filename + ".png")
        fig.show()
        return 
    else: 
        return plot_data

def get_blade_coordinates(rotor,n_points,dim,i,aircraftRefFrame=True):
    """ This generates the coordinates of the blade surface for plotting in the aircraft frame (x-back, z-up)

    Assumptions:
    None

    Source:
    None

    Inputs:
    prop             - SUAVE rotor
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

def contour_surface_slice(x,y,z,values,color_scale):
    return go.Surface(x=x,y=y,z=z,surfacecolor=values,colorscale=color_scale, showscale=False)
 