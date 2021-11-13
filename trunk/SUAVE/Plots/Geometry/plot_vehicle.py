## @ingroup Plots-Geometry
# plot_vehicle.py
#
# Created:  Mar 2020, M. Clarke
#           Apr 2020, M. Clarke
#           Jul 2020, M. Clarke
#           Jul 2021, E. Botero
#           Oct 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry import import_airfoil_geometry
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_naca_4series import compute_naca_4series
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.generate_vortex_distribution  import generate_vortex_distribution 
from SUAVE.Analyses.Aerodynamics import Vortex_Lattice

## @ingroup Plots-Geometry
def plot_vehicle(vehicle, elevation_angle = 30,azimuthal_angle = 210, axis_limits = 10,
                 save_figure = False, plot_control_points = True, save_filename = "Vehicle_Geometry"):
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

    # initalize figure
    fig = plt.figure(save_filename)
    fig.set_size_inches(8,8)
    axes = plt.axes(projection='3d')
    axes.view_init(elev= elevation_angle, azim= azimuthal_angle)

    # -------------------------------------------------------------------------
    # PLOT WING
    # -------------------------------------------------------------------------
    wing_face_color = 'darkgrey'
    wing_edge_color = 'grey'
    wing_alpha_val  = 1
    plot_wing(axes,VD,wing_face_color,wing_edge_color,wing_alpha_val)
    if  plot_control_points:
        axes.scatter(VD.XC,VD.YC,VD.ZC, c='r', marker = 'o' )

    # -------------------------------------------------------------------------
    # PLOT WAKE
    # -------------------------------------------------------------------------
    wake_face_color = 'white'
    wake_edge_color = 'blue'
    wake_alpha      = 0.5
    if'Wake' in VD:
        plot_propeller_wake(axes, VD,wake_face_color,wake_edge_color,wake_alpha)

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
        plot_fuselage_geometry(axes,fus_pts,fuselage_face_color,fuselage_edge_color,fuselage_alpha)

    # -------------------------------------------------------------------------
    # PLOT ENGINE
    # -------------------------------------------------------------------------
    nacelle_face_color = 'darkred'
    nacelle_edge_color = 'black'
    nacelle_alpha      = 1
    for nacelle in vehicle.nacelles:  
        # Generate Nacelle Geoemtry
        nac_geo = generate_nacelle_points(nacelle)
        
        # Plot Nacelle Geometry
        plot_nacelle_geometry(axes,nac_geo,nacelle_face_color,nacelle_edge_color,nacelle_alpha ) 

           
    # -------------------------------------------------------------------------
    # PLOT ENGINE
    # -------------------------------------------------------------------------
    network_face_color = 'darkred'
    network_edge_color = 'red'
    network_alpha      = 1
    for network in vehicle.networks:
        plot_network(axes,network,network_face_color,network_edge_color,network_alpha )

    axes.set_xlim(0,axis_limits*2)
    axes.set_ylim(-axis_limits,axis_limits)
    axes.set_zlim(-axis_limits,axis_limits)

    plt.axis('off')
    plt.grid(None)
    return

def plot_wing(axes,VD,face_color,edge_color,alpha_val):
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

        X = [VD.XA1[i],VD.XB1[i],VD.XB2[i],VD.XA2[i]]
        Y = [VD.YA1[i],VD.YB1[i],VD.YB2[i],VD.YA2[i]]
        Z = [VD.ZA1[i],VD.ZB1[i],VD.ZB2[i],VD.ZA2[i]]

        verts = [list(zip(X, Y, Z))]

        collection = Poly3DCollection(verts)
        collection.set_facecolor(face_color)
        collection.set_edgecolor(edge_color)
        collection.set_alpha(alpha_val)

        axes.add_collection3d(collection)

    return

def plot_propeller_wake(axes, VD,face_color,edge_color,alpha):
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
    num_prop = len(VD.Wake.XA1[0,:,0,0,0])
    num_B    = len(VD.Wake.XA1[0,0,:,0,0])
    dim_R    = len(VD.Wake.XA1[0,0,0,:,0])
    nts      = len(VD.Wake.XA1[0,0,0,0,:])
    for p_idx in range(num_prop):
        for t_idx in range(nts):
            for B_idx in range(num_B):
                for loc in range(dim_R):
                    X = [VD.Wake.XA1[0,p_idx,B_idx,loc,t_idx],
                         VD.Wake.XB1[0,p_idx,B_idx,loc,t_idx],
                         VD.Wake.XB2[0,p_idx,B_idx,loc,t_idx],
                         VD.Wake.XA2[0,p_idx,B_idx,loc,t_idx]]
                    Y = [VD.Wake.YA1[0,p_idx,B_idx,loc,t_idx],
                         VD.Wake.YB1[0,p_idx,B_idx,loc,t_idx],
                         VD.Wake.YB2[0,p_idx,B_idx,loc,t_idx],
                         VD.Wake.YA2[0,p_idx,B_idx,loc,t_idx]]
                    Z = [VD.Wake.ZA1[0,p_idx,B_idx,loc,t_idx],
                         VD.Wake.ZB1[0,p_idx,B_idx,loc,t_idx],
                         VD.Wake.ZB2[0,p_idx,B_idx,loc,t_idx],
                         VD.Wake.ZA2[0,p_idx,B_idx,loc,t_idx]]
                    verts = [list(zip(X, Y, Z))]
                    collection = Poly3DCollection(verts)
                    collection.set_facecolor(face_color)
                    collection.set_edgecolor(edge_color)
                    collection.set_alpha(alpha)
                    axes.add_collection3d(collection)
    return


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


def plot_fuselage_geometry(axes,fus_pts, face_color,edge_color,alpha):
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
                X = [fus_pts[i_seg  ,i_tes  ,0],
                     fus_pts[i_seg  ,i_tes+1,0],
                     fus_pts[i_seg+1,i_tes+1,0],
                     fus_pts[i_seg+1,i_tes  ,0]]
                Y = [fus_pts[i_seg  ,i_tes  ,1],
                     fus_pts[i_seg  ,i_tes+1,1],
                     fus_pts[i_seg+1,i_tes+1,1],
                     fus_pts[i_seg+1,i_tes  ,1]]
                Z = [fus_pts[i_seg  ,i_tes  ,2],
                     fus_pts[i_seg  ,i_tes+1,2],
                     fus_pts[i_seg+1,i_tes+1,2],
                     fus_pts[i_seg+1,i_tes  ,2]]
                verts = [list(zip(X, Y, Z))]
                collection = Poly3DCollection(verts)
                collection.set_facecolor(face_color)
                collection.set_edgecolor(edge_color)
                collection.set_alpha(alpha)
                axes.add_collection3d(collection)

    return


def plot_network(axes,network,prop_face_color,prop_edge_color,prop_alpha):
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

    if ('propellers' in network.keys()):

        for prop in network.propellers:

            # Generate And Plot Propeller/Rotor Geometry
            plot_propeller_geometry(axes,prop,network,'propeller',prop_face_color,prop_edge_color,prop_alpha)

    if ('lift_rotors' in network.keys()):

        for rotor in network.lift_rotors:

            # Generate and Plot Propeller/Rotor Geometry
            plot_propeller_geometry(axes,rotor,network,'lift_rotor',prop_face_color,prop_edge_color,prop_alpha)

    return 

def generate_nacelle_points(nac,tessellation = 24):
    """ This generates the coordinate points on the surface of the nacelle

    Assumptions:
    None

    Source:
    None

    Inputs: 
    Properties Used:
    N/A 
    """
     
    
    num_nac_segs = len(nac.Segments.keys())   
    theta        = np.linspace(0,2*np.pi,tessellation)
    n_points     = 20
    
    if num_nac_segs == 0:
        num_nac_segs = int(n_points/2)
        nac_pts      = np.zeros((num_nac_segs,tessellation,3))
        naf          = nac.Airfoil
        
        if naf.naca_4_series_airfoil != None: 
            # use mean camber surface of airfoil
            camber       = float(naf.naca_4_series_airfoil[0])/100
            camber_loc   = float(naf.naca_4_series_airfoil[1])/10
            thickness    = float(naf.naca_4_series_airfoil[2:])/100 
            airfoil_data = compute_naca_4series(camber, camber_loc, thickness,(n_points - 2))
            xpts         = np.repeat(np.atleast_2d(airfoil_data.x_lower_surface).T,tessellation,axis = 1)*nac.length 
            zpts         = np.repeat(np.atleast_2d(airfoil_data.camber_coordinates[0]).T,tessellation,axis = 1)*nac.length  
        
        elif naf.coordinate_file != None: 
            a_sec        = naf.coordinate_file
            a_secl       = [0]
            airfoil_data = import_airfoil_geometry(a_sec,npoints=num_nac_segs)
            xpts         = np.repeat(np.atleast_2d(np.take(airfoil_data.x_coordinates,a_secl,axis=0)).T,tessellation,axis = 1)*nac.length  
            zpts         = np.repeat(np.atleast_2d(np.take(airfoil_data.y_coordinates,a_secl,axis=0)).T,tessellation,axis = 1)*nac.length  
        
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

def plot_nacelle_geometry(axes,NAC_SURF_PTS,face_color,edge_color,alpha):
    """ This plots a 3D surface of a nacelle  

    Assumptions:
    None

    Source:
    None 

    Properties Used:
    N/A
    """

    num_nac_segs = len(NAC_SURF_PTS[:,0,0])
    tesselation  = len(NAC_SURF_PTS[0,:,0]) 
    for i_seg in range(num_nac_segs-1):
        for i_tes in range(tesselation-1):
            X = [NAC_SURF_PTS[i_seg  ,i_tes  ,0],
                 NAC_SURF_PTS[i_seg  ,i_tes+1,0],
                 NAC_SURF_PTS[i_seg+1,i_tes+1,0],
                 NAC_SURF_PTS[i_seg+1,i_tes  ,0]]
            Y = [NAC_SURF_PTS[i_seg  ,i_tes  ,1],
                 NAC_SURF_PTS[i_seg  ,i_tes+1,1],
                 NAC_SURF_PTS[i_seg+1,i_tes+1,1],
                 NAC_SURF_PTS[i_seg+1,i_tes  ,1]]
            Z = [NAC_SURF_PTS[i_seg  ,i_tes  ,2],
                 NAC_SURF_PTS[i_seg  ,i_tes+1,2],
                 NAC_SURF_PTS[i_seg+1,i_tes+1,2],
                 NAC_SURF_PTS[i_seg+1,i_tes  ,2]]
            verts = [list(zip(X, Y, Z))]
            collection = Poly3DCollection(verts)
            collection.set_facecolor(face_color)
            collection.set_edgecolor(edge_color)
            collection.set_alpha(alpha)
            axes.add_collection3d(collection)

    return

def plot_propeller_geometry(axes,prop,network,network_name,prop_face_color='red',prop_edge_color='darkred',prop_alpha=1):
    """ This plots a 3D surface of the  propeller

    Assumptions:
    None

    Source:
    None

    Inputs:
    network            - network data structure

    Properties Used:
    N/A
    """

    # unpack
    num_B  = prop.number_of_blades
    a_sec  = prop.airfoil_geometry
    a_secl = prop.airfoil_polar_stations
    beta   = prop.twist_distribution
    a_o    = prop.azimuthal_offset_angle
    b      = prop.chord_distribution
    r      = prop.radius_distribution
    MCA    = prop.mid_chord_alignment
    t      = prop.max_thickness_distribution
    origin = prop.origin
 
    n_points  = 20
    af_pts    = n_points-1
    dim       = len(b)
    theta     = np.linspace(0,2*np.pi,num_B+1)[:-1]

    # create empty data structure for storing geometry
    G = Data()


    rot    = prop.rotation 
    flip_1 =  (np.pi/2)
    flip_2 =  (np.pi/2)

    MCA_2d = np.repeat(np.atleast_2d(MCA).T,n_points,axis=1)
    b_2d   = np.repeat(np.atleast_2d(b).T  ,n_points,axis=1)
    t_2d   = np.repeat(np.atleast_2d(t).T  ,n_points,axis=1)
    r_2d   = np.repeat(np.atleast_2d(r).T  ,n_points,axis=1)
    shift  = np.repeat(np.atleast_2d(np.ones_like(b)*b[0]).T  ,n_points,axis=1)

    for i in range(num_B):
        # get airfoil coordinate geometry
        if a_sec != None:
            airfoil_data = import_airfoil_geometry(a_sec,npoints=n_points)
            xpts         = np.take(airfoil_data.x_coordinates,a_secl,axis=0)
            zpts         = np.take(airfoil_data.y_coordinates,a_secl,axis=0)
            max_t        = np.take(airfoil_data.thickness_to_chord,a_secl,axis=0)

        else:
            camber       = 0.02
            camber_loc   = 0.4
            thickness    = 0.10
            airfoil_data = compute_naca_4series(camber, camber_loc, thickness,(n_points - 2))
            xpts         = np.repeat(np.atleast_2d(airfoil_data.x_coordinates) ,dim,axis=0)
            zpts         = np.repeat(np.atleast_2d(airfoil_data.y_coordinates) ,dim,axis=0)
            max_t        = np.repeat(airfoil_data.thickness_to_chord,dim,axis=0)

        # store points of airfoil in similar format as Vortex Points (i.e. in vertices)
        max_t2d = np.repeat(np.atleast_2d(max_t).T ,n_points,axis=1)

        xp      = (- MCA_2d + xpts*b_2d - shift/2 )     # x-coord of airfoil
        yp      = r_2d*np.ones_like(xp)       # radial location
        zp      = zpts*(t_2d/max_t2d)         # former airfoil y coord

        matrix = np.zeros((len(zp),n_points,3)) # radial location, airfoil pts (same y)
        matrix[:,:,0] = xp*rot
        matrix[:,:,1] = yp
        matrix[:,:,2] = zp

        # ROTATION MATRICES FOR INNER SECTION
        # rotation about y axis to create twist and position blade upright
        trans_1 = np.zeros((dim,3,3))
        trans_1[:,0,0] = np.cos(flip_1 - rot*beta)
        trans_1[:,0,2] = -np.sin(flip_1 - rot*beta)
        trans_1[:,1,1] = 1
        trans_1[:,2,0] = np.sin(flip_1 - rot*beta)
        trans_1[:,2,2] = np.cos(flip_1 - rot*beta)

        # rotation about x axis to create azimuth locations
        trans_2 = np.array([[1 , 0 , 0],
                       [0 , np.cos(theta[i] + a_o + flip_2 ), -np.sin(theta[i] +a_o +  flip_2)],
                       [0,np.sin(theta[i] + a_o + flip_2), np.cos(theta[i] + a_o + flip_2)]])
        trans_2 =  np.repeat(trans_2[ np.newaxis,:,: ],dim,axis=0)

        # rotation about y to orient propeller/rotor to thrust angle
        trans_3 =  prop.prop_vel_to_body()
        trans_3 =  np.repeat(trans_3[ np.newaxis,:,: ],dim,axis=0) 
        
        # rotation 180 degrees 
        trans_4 = np.zeros((dim,3,3))
        trans_4[:,0,0] = np.cos(np.pi)
        trans_4[:,0,2] = -np.sin(np.pi)
        trans_4[:,1,1] = 1
        trans_4[:,2,0] = np.sin(np.pi)
        trans_4[:,2,2] = np.cos(np.pi)
        
        trans     = np.matmul(trans_4,np.matmul(trans_3,np.matmul(trans_2,trans_1)))
        rot_mat   = np.repeat(trans[:, np.newaxis,:,:],n_points,axis=1)

        # ---------------------------------------------------------------------------------------------
        # ROTATE POINTS
        mat  =  np.matmul(rot_mat,matrix[...,None]).squeeze()

        # ---------------------------------------------------------------------------------------------
        # store points
        G.XA1  = mat[:-1,:-1,0] + origin[0][0]
        G.YA1  = mat[:-1,:-1,1] + origin[0][1]
        G.ZA1  = mat[:-1,:-1,2] + origin[0][2]
        G.XA2  = mat[:-1,1:,0]  + origin[0][0]
        G.YA2  = mat[:-1,1:,1]  + origin[0][1]
        G.ZA2  = mat[:-1,1:,2]  + origin[0][2]

        G.XB1  = mat[1:,:-1,0] + origin[0][0]
        G.YB1  = mat[1:,:-1,1] + origin[0][1]
        G.ZB1  = mat[1:,:-1,2] + origin[0][2]
        G.XB2  = mat[1:,1:,0]  + origin[0][0]
        G.YB2  = mat[1:,1:,1]  + origin[0][1]
        G.ZB2  = mat[1:,1:,2]  + origin[0][2]

        # ------------------------------------------------------------------------
        # Plot Propeller Blade
        # ------------------------------------------------------------------------
        for sec in range(dim-1):
            for loc in range(af_pts):
                X = [G.XA1[sec,loc],
                     G.XB1[sec,loc],
                     G.XB2[sec,loc],
                     G.XA2[sec,loc]]
                Y = [G.YA1[sec,loc],
                     G.YB1[sec,loc],
                     G.YB2[sec,loc],
                     G.YA2[sec,loc]]
                Z = [G.ZA1[sec,loc],
                     G.ZB1[sec,loc],
                     G.ZB2[sec,loc],
                     G.ZA2[sec,loc]]
                prop_verts = [list(zip(X, Y, Z))]
                prop_collection = Poly3DCollection(prop_verts)
                prop_collection.set_facecolor(prop_face_color)
                prop_collection.set_edgecolor(prop_edge_color)
                prop_collection.set_alpha(prop_alpha)
                axes.add_collection3d(prop_collection)
    return
