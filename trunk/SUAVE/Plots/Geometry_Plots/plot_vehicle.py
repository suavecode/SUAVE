## @ingroup Plots-Geometry_Plots
# plot_vehicle.py
# 
# Created:  Mar 2020, M. Clarke
# Modified: Apr 2020, M. Clarke
#           Jul 2020, M. Clarke
#           Jun 2021, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------  
from SUAVE.Core import Data
import numpy as np 
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry import import_airfoil_geometry
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_naca_4series import compute_naca_4series  
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.generate_wing_vortex_distribution  import generate_wing_vortex_distribution
from SUAVE.Components.Energy.Networks import Lift_Cruise 

## @ingroup Plots-Geometry_Plots
def plot_vehicle(vehicle, save_figure = False, plot_control_points = True, save_filename = "Vehicle_Geometry"):     
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
    # unpack vortex distribution 
    try:
        VD = vehicle.vortex_distribution 
    except:
        settings = Data()
        settings.number_spanwise_vortices  = 25
        settings.number_chordwise_vortices = 5
        settings.spanwise_cosine_spacing   = False 
        settings.model_fuselage            = False
        VD = generate_wing_vortex_distribution(vehicle,settings)  
        
    # initalize figure 
    fig = plt.figure(save_filename) 
    fig.set_size_inches(8,8) 
    axes = Axes3D(fig)    
    axes.view_init(elev= 30, azim= 210)  
    
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
        fus_pts = generate_fuselage_points(axes, fus) 
        
        # Plot Fuselage Geometry          
        plot_fuselage_geometry(axes,fus_pts,fuselage_face_color,fuselage_edge_color,fuselage_alpha)  
    # -------------------------------------------------------------------------
    # PLOT ENGINE
    # -------------------------------------------------------------------------        
    propulsor_face_color = 'darkred'                
    propulsor_edge_color = 'black' 
    propulsor_alpha      = 1    
    for propulsor in vehicle.propulsors:    
        plot_propulsor(axes,propulsor)    
      
    # Plot Vehicle
    plt.axis('off') 
    plt.grid(None)      
    if save_figure:
        plt.savefig(save_filename)
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
        max_range = np.array([VD.X.max()-VD.X.min(), VD.Y.max()-VD.Y.min(), VD.Z.max()-VD.Z.min()]).max() / 2.0 
        
        mid_x = (VD.X .max()+VD.X .min()) * 0.5
        mid_y = (VD.Y .max()+VD.Y .min()) * 0.5
        mid_z = (VD.Z .max()+VD.Z .min()) * 0.5
        
        axes.set_xlim(mid_x - max_range, mid_x + max_range)
        axes.set_ylim(mid_y - max_range, mid_y + max_range)
        axes.set_zlim(mid_z - max_range, mid_z + max_range)    
        
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
    num_prop = len(VD.Wake.XA1[:,0,0,0])
    nts      = len(VD.Wake.XA1[0,:,0,0])
    num_B    = len(VD.Wake.XA1[0,0,:,0])
    dim_R    = len(VD.Wake.XA1[0,0,0,:])
    for p_idx in range(num_prop): 
        for t_idx in range(nts): 
            for B_idx in range(num_B):
                for loc in range(dim_R): 
                    X = [VD.Wake.XA1[p_idx,t_idx,B_idx,loc],
                         VD.Wake.XB1[p_idx,t_idx,B_idx,loc],
                         VD.Wake.XB2[p_idx,t_idx,B_idx,loc],
                         VD.Wake.XA2[p_idx,t_idx,B_idx,loc]]
                    Y = [VD.Wake.YA1[p_idx,t_idx,B_idx,loc],
                         VD.Wake.YB1[p_idx,t_idx,B_idx,loc],
                         VD.Wake.YB2[p_idx,t_idx,B_idx,loc],
                         VD.Wake.YA2[p_idx,t_idx,B_idx,loc]]
                    Z = [VD.Wake.ZA1[p_idx,t_idx,B_idx,loc],
                         VD.Wake.ZB1[p_idx,t_idx,B_idx,loc],
                         VD.Wake.ZB2[p_idx,t_idx,B_idx,loc],
                         VD.Wake.ZA2[p_idx,t_idx,B_idx,loc]]                    
                    verts = [list(zip(X, Y, Z))]
                    collection = Poly3DCollection(verts)
                    collection.set_facecolor(face_color)
                    collection.set_edgecolor(edge_color) 
                    collection.set_alpha(alpha)
                    axes.add_collection3d(collection)  
    return 
    

def generate_fuselage_points(axes, fus ,tessellation = 24 ):
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


def plot_propulsor(axes,propulsor):  
    """ This plots the 3D surface of the propulsor

    Assumptions: 
    None

    Source:   
    None
    
    Inputs:     
    propulsor            - propulsor data structure
    propulsor_face_color - color of panel
    propulsor_edge_color - color of panel edge
    propulsor_alpha      - translucency:  1 = opaque , 0 = transparent 
    
    Properties Used:
    N/A
    """          
    
    if ('propeller' in propulsor.keys()):  
        prop = propulsor.propeller 
        try:
            propulsor.thrust_angle = propulsor.propeller_thrust_angle     
        except:
            pass 
        
        # Generate And Plot Propeller/Rotor Geometry   
        plot_propeller_geometry(axes,prop,propulsor,'propeller') 
        
    if ('rotor' in propulsor.keys()):  
        rot = propulsor.rotor   
        try:
            propulsor.thrust_angle = propulsor.rotor_thrust_angle     
        except:
            pass
        
        # Generate and Plot Propeller/Rotor Geometry   
        plot_propeller_geometry(axes,rot,propulsor,'rotor')        
    
    return 

def plot_propeller_geometry(axes,prop,propulsor,propulsor_name):
    """ This plots a 3D surface of the  propeller

    Assumptions: 
    None

    Source:   
    None
    
    Inputs:    
    propulsor            - propulsor data structure 
    
    Properties Used:
    N/A
    """          
        
    # unpack         
    num_B  = prop.number_of_blades      
    a_sec  = prop.airfoil_geometry          
    a_secl = prop.airfoil_polar_stations
    beta   = prop.twist_distribution         
    b      = prop.chord_distribution         
    r      = prop.radius_distribution 
    MCA    = prop.mid_chord_alignment
    t      = prop.max_thickness_distribution
    ta     = -propulsor.thrust_angle
    try:
        a_o = -prop.azimuthal_offset
    except:
        a_o = 0.0 # no offset
    
    if isinstance(propulsor,Lift_Cruise):
        if propulsor_name == 'propeller': 
            origin = propulsor.propeller.origin
            
        elif propulsor_name == 'rotor': 
            origin = propulsor.rotor.origin
    else:
        origin = prop.origin
    n_points  = 10
    af_pts    = (2*n_points)-1
    dim       = len(b)
    dim2      = 2*n_points
    num_props = len(origin) 
    theta     = np.linspace(0,2*np.pi,num_B+1)[:-1]   
    
    # create empty data structure for storing geometry
    G = Data()    
    
    for n_p in range(num_props):  
        rot    = prop.rotation[n_p] 
        a_o    = a_o*rot
        flip_1 = (np.pi/2)  
        flip_2 = (np.pi/2)  
        
        MCA_2d = np.repeat(np.atleast_2d(MCA).T,dim2,axis=1)
        b_2d   = np.repeat(np.atleast_2d(b).T  ,dim2,axis=1)
        t_2d   = np.repeat(np.atleast_2d(t).T  ,dim2,axis=1)
        r_2d   = np.repeat(np.atleast_2d(r).T  ,dim2,axis=1)
        
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
                airfoil_data = compute_naca_4series(camber, camber_loc, thickness,(n_points*2 - 2))                  
                xpts         = np.repeat(np.atleast_2d(airfoil_data.x_coordinates) ,dim,axis=0)
                zpts         = np.repeat(np.atleast_2d(airfoil_data.y_coordinates) ,dim,axis=0)
                max_t        = np.repeat(airfoil_data.thickness_to_chord,dim,axis=0) 
             
            # store points of airfoil in similar format as Vortex Points (i.e. in vertices)   
            max_t2d = np.repeat(np.atleast_2d(max_t).T ,dim2,axis=1)
            
            xp      = rot*(- MCA_2d + xpts*b_2d)  # x coord of airfoil
            yp      = r_2d*np.ones_like(xp)       # radial location        
            zp      = zpts*(t_2d/max_t2d)         # former airfoil y coord 
                              
            matrix = np.zeros((len(zp),dim2,3)) # radial location, airfoil pts (same y)   
            matrix[:,:,0] = xp
            matrix[:,:,1] = yp
            matrix[:,:,2] = zp
            
            # ROTATION MATRICES FOR INNER SECTION     
            # rotation about y axis to create twist and position blade upright  
            trans_1 = np.zeros((dim,3,3))
            trans_1[:,0,0] = np.cos(rot*flip_1 - rot*beta)           
            trans_1[:,0,2] = -np.sin(rot*flip_1 - rot*beta)                 
            trans_1[:,1,1] = 1
            trans_1[:,2,0] = np.sin(rot*flip_1 - rot*beta) 
            trans_1[:,2,2] = np.cos(rot*flip_1 - rot*beta) 
    
            # rotation about x axis to create azimuth locations 
            trans_2 = np.array([[1 , 0 , 0],
                           [0 , np.cos(theta[i] + rot*a_o + flip_2 ), -np.sin(theta[i] + rot*a_o + flip_2)],
                           [0,np.sin(theta[i] + rot*a_o + flip_2), np.cos(theta[i] + rot*a_o + flip_2)]]) 
            trans_2 =  np.repeat(trans_2[ np.newaxis,:,: ],dim,axis=0)
            
            # roation about y to orient propeller/rotor to thrust angle 
            trans_3 = np.array([[np.cos(ta),0 , -np.sin(ta)],
                           [0 ,  1 , 0] ,
                           [np.sin(ta) , 0 , np.cos(ta)]])
            trans_3 =  np.repeat(trans_3[ np.newaxis,:,: ],dim,axis=0)
            
            trans     = np.matmul(trans_3,np.matmul(trans_2,trans_1))
            rot_mat   = np.repeat(trans[:, np.newaxis,:,:],dim2,axis=1)
             
            # ---------------------------------------------------------------------------------------------
            # ROTATE POINTS
            mat  =  np.matmul(rot_mat,matrix[...,None]).squeeze() 
            
            # ---------------------------------------------------------------------------------------------
            # store points
            G.XA1  = mat[:-1,:-1,0] + origin[n_p][0]
            G.YA1  = mat[:-1,:-1,1] + origin[n_p][1] 
            G.ZA1  = mat[:-1,:-1,2] + origin[n_p][2]
            G.XA2  = mat[:-1,1:,0]  + origin[n_p][0]
            G.YA2  = mat[:-1,1:,1]  + origin[n_p][1] 
            G.ZA2  = mat[:-1,1:,2]  + origin[n_p][2]
                            
            G.XB1  = mat[1:,:-1,0] + origin[n_p][0]
            G.YB1  = mat[1:,:-1,1] + origin[n_p][1]  
            G.ZB1  = mat[1:,:-1,2] + origin[n_p][2]
            G.XB2  = mat[1:,1:,0]  + origin[n_p][0]
            G.YB2  = mat[1:,1:,1]  + origin[n_p][1]
            G.ZB2  = mat[1:,1:,2]  + origin[n_p][2]    
             
            # ------------------------------------------------------------------------
            # Plot Propeller Blade 
            # ------------------------------------------------------------------------
            prop_face_color = 'red'
            prop_edge_color = 'red'
            prop_alpha      = 1
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

 
def generate_wing_points(vehicle,settings):
    ''' 
        _ts  = true surface 
    '''
    # ---------------------------------------------------------------------------------------
    # STEP 1: Define empty vectors for coordinates of panes, control points and bound vortices
    # ---------------------------------------------------------------------------------------
    VD = Data()
 
    VD.XA1         = np.empty(shape=[0,1])
    VD.YA1         = np.empty(shape=[0,1])  
    VD.ZA1         = np.empty(shape=[0,1])
    VD.XA2         = np.empty(shape=[0,1])
    VD.YA2         = np.empty(shape=[0,1])    
    VD.ZA2         = np.empty(shape=[0,1])    
    VD.XB1         = np.empty(shape=[0,1])
    VD.YB1         = np.empty(shape=[0,1])  
    VD.ZB1         = np.empty(shape=[0,1])
    VD.XB2         = np.empty(shape=[0,1])
    VD.YB2         = np.empty(shape=[0,1])    
    VD.ZB2         = np.empty(shape=[0,1])  
    VD.XA1_ts      = np.empty(shape=[0,1])
    VD.YA1_ts      = np.empty(shape=[0,1])  
    VD.ZA1_ts      = np.empty(shape=[0,1])
    VD.XA2_ts      = np.empty(shape=[0,1])
    VD.YA2_ts      = np.empty(shape=[0,1])    
    VD.ZA2_ts      = np.empty(shape=[0,1])    
    VD.XB1_ts      = np.empty(shape=[0,1])
    VD.YB1_ts      = np.empty(shape=[0,1])  
    VD.ZB1_ts      = np.empty(shape=[0,1])
    VD.XB2_ts      = np.empty(shape=[0,1])
    VD.YB2_ts      = np.empty(shape=[0,1])    
    VD.ZB2_ts      = np.empty(shape=[0,1])  
    VD.XC          = np.empty(shape=[0,1])
    VD.YC          = np.empty(shape=[0,1])
    VD.ZC          = np.empty(shape=[0,1])     
    n_sw           = settings.number_spanwise_vortices 
    n_cw           = settings.number_chordwise_vortices     
    spc            = settings.spanwise_cosine_spacing 

    # ---------------------------------------------------------------------------------------
    # STEP 2: Unpack aircraft wings  
    # ---------------------------------------------------------------------------------------    
    n_w         = 0  # instantiate the number of wings counter  
    n_cp        = 0  # instantiate number of bound vortices counter     
    n_sp        = 0  # instantiate number of true surface panels    
    wing_areas  = [] # instantiate wing areas
    vortex_lift = []
    
    for wing in vehicle.wings: 
        span          = wing.spans.projected
        root_chord    = wing.chords.root
        tip_chord     = wing.chords.tip
        sweep_qc      = wing.sweeps.quarter_chord
        sweep_le      = wing.sweeps.leading_edge 
        twist_rc      = wing.twists.root
        twist_tc      = wing.twists.tip
        dihedral      = wing.dihedral
        sym_para      = wing.symmetric 
        vertical_wing = wing.vertical
        wing_origin   = wing.origin[0]
        vortex_lift.append(wing.vortex_lift)
        
        # determine if vehicle has symmetry 
        if sym_para is True :
            span = span/2
            vortex_lift.append(wing.vortex_lift)
        
        if spc == True:
            
            # discretize wing using cosine spacing
            n               = np.linspace(n_sw+1,0,n_sw+1)         # vectorize
            thetan          = n*(np.pi/2)/(n_sw+1)                 # angular stations
            y_coordinates   = span*np.cos(thetan)                  # y locations based on the angular spacing
        else:
        
            # discretize wing using linear spacing
            y_coordinates   = np.linspace(0,span,n_sw+1) 
        
        # create empty vectors for coordinates 
        xa1        = np.zeros(n_cw*n_sw)
        ya1        = np.zeros(n_cw*n_sw)
        za1        = np.zeros(n_cw*n_sw)
        xa2        = np.zeros(n_cw*n_sw)
        ya2        = np.zeros(n_cw*n_sw)
        za2        = np.zeros(n_cw*n_sw)    
        xb1        = np.zeros(n_cw*n_sw)
        yb1        = np.zeros(n_cw*n_sw)
        zb1        = np.zeros(n_cw*n_sw)
        xb2        = np.zeros(n_cw*n_sw) 
        yb2        = np.zeros(n_cw*n_sw) 
        zb2        = np.zeros(n_cw*n_sw)  
        xa1_ts     = np.zeros(2*n_cw*n_sw)
        ya1_ts     = np.zeros(2*n_cw*n_sw)
        za1_ts     = np.zeros(2*n_cw*n_sw)
        xa2_ts     = np.zeros(2*n_cw*n_sw)
        ya2_ts     = np.zeros(2*n_cw*n_sw)
        za2_ts     = np.zeros(2*n_cw*n_sw)    
        xb1_ts     = np.zeros(2*n_cw*n_sw)
        yb1_ts     = np.zeros(2*n_cw*n_sw)
        zb1_ts     = np.zeros(2*n_cw*n_sw)
        xb2_ts     = np.zeros(2*n_cw*n_sw) 
        yb2_ts     = np.zeros(2*n_cw*n_sw) 
        zb2_ts     = np.zeros(2*n_cw*n_sw)           
        xc         = np.zeros(n_cw*n_sw) 
        yc         = np.zeros(n_cw*n_sw) 
        zc         = np.zeros(n_cw*n_sw)         
        cs_w       = np.zeros(n_sw)

        # ---------------------------------------------------------------------------------------
        # STEP 3: Determine if wing segments are defined  
        # ---------------------------------------------------------------------------------------
        n_segments           = len(wing.Segments.keys())
        if n_segments>0:            
            # ---------------------------------------------------------------------------------------
            # STEP 4A: Discretizing the wing sections into panels
            # ---------------------------------------------------------------------------------------
            segment_chord          = np.zeros(n_segments)
            segment_twist          = np.zeros(n_segments)
            segment_sweep          = np.zeros(n_segments)
            segment_span           = np.zeros(n_segments)
            segment_area           = np.zeros(n_segments)
            segment_dihedral       = np.zeros(n_segments)
            segment_x_coord        = [] 
            segment_camber         = []
            segment_top_surface    = []
            segment_bot_surface    = []
            segment_chord_x_offset = np.zeros(n_segments)
            segment_chord_z_offset = np.zeros(n_segments)
            section_stations       = np.zeros(n_segments) 

            # ---------------------------------------------------------------------------------------
            # STEP 5A: Obtain sweep, chord, dihedral and twist at the beginning/end of each segment.
            #          If applicable, append airfoil section VD and flap/aileron deflection angles.
            # --------------------------------------------------------------------------------------- 
            for i_seg in range(n_segments):   
                segment_chord[i_seg]    = wing.Segments[i_seg].root_chord_percent*root_chord
                segment_twist[i_seg]    = wing.Segments[i_seg].twist
                section_stations[i_seg] = wing.Segments[i_seg].percent_span_location*span  
                segment_dihedral[i_seg] = wing.Segments[i_seg].dihedral_outboard                    

                # change to leading edge sweep, if quarter chord sweep givent, convert to leading edge sweep 
                if (i_seg == n_segments-1):
                    segment_sweep[i_seg] = 0                                  
                else: 
                    if wing.Segments[i_seg].sweeps.leading_edge != None:
                        segment_sweep[i_seg] = wing.Segments[i_seg].sweeps.leading_edge
                    else:                                                                 
                        sweep_quarter_chord  = wing.Segments[i_seg].sweeps.quarter_chord
                        cf       = 0.25                          
                        seg_root_chord       = root_chord*wing.Segments[i_seg].root_chord_percent
                        seg_tip_chord        = root_chord*wing.Segments[i_seg+1].root_chord_percent
                        seg_span             = span*(wing.Segments[i_seg+1].percent_span_location - wing.Segments[i_seg].percent_span_location )
                        segment_sweep[i_seg] = np.arctan(((seg_root_chord*cf) + (np.tan(sweep_quarter_chord)*seg_span - cf*seg_tip_chord)) /seg_span)  

                if i_seg == 0:
                    segment_span[i_seg]           = 0.0
                    segment_chord_x_offset[i_seg] = 0.0  
                    segment_chord_z_offset[i_seg] = 0.0       
                else:
                    segment_span[i_seg]           = wing.Segments[i_seg].percent_span_location*span - wing.Segments[i_seg-1].percent_span_location*span
                    segment_chord_x_offset[i_seg] = segment_chord_x_offset[i_seg-1] + segment_span[i_seg]*np.tan(segment_sweep[i_seg-1])
                    segment_chord_z_offset[i_seg] = segment_chord_z_offset[i_seg-1] + segment_span[i_seg]*np.tan(segment_dihedral[i_seg-1])
                    segment_area[i_seg]           = 0.5*(root_chord*wing.Segments[i_seg-1].root_chord_percent + root_chord*wing.Segments[i_seg].root_chord_percent)*segment_span[i_seg]

                # Get airfoil section VD  
                if wing.Segments[i_seg].Airfoil: 
                    airfoil_data = import_airfoil_geometry([wing.Segments[i_seg].Airfoil.airfoil.coordinate_file])    
                    segment_camber.append(airfoil_data.camber_coordinates[0])
                    segment_top_surface.append(airfoil_data.y_upper_surface[0])
                    segment_bot_surface.append(airfoil_data.y_lower_surface[0])
                    segment_x_coord.append(airfoil_data.x_lower_surface[0]) 
                else:
                    dummy_dimension  = 30 
                    segment_camber.append(np.zeros(dummy_dimension))   
                    airfoil_data = compute_naca_4series(0.0, 0.0,wing.thickness_to_chord,dummy_dimension*2 - 2)
                    segment_top_surface.append(airfoil_data.y_upper_surface[0])
                    segment_bot_surface.append(airfoil_data.y_lower_surface[0])                    
                    segment_x_coord.append(np.linspace(0,1,30))  

            wing_areas.append(np.sum(segment_area[:]))
            if sym_para is True :
                wing_areas.append(np.sum(segment_area[:]))            

            # Shift spanwise vortices onto section breaks  
            if len(y_coordinates) < n_segments:
                raise ValueError('Not enough spanwise VLM stations for segment breaks')

            last_idx = None            
            for i_seg in range(n_segments):
                idx =  (np.abs(y_coordinates-section_stations[i_seg])).argmin()
                if last_idx is not None and idx <= last_idx:
                    idx = last_idx + 1
                y_coordinates[idx] = section_stations[i_seg]   
                last_idx = idx


            for i_seg in range(n_segments):
                if section_stations[i_seg] not in y_coordinates:
                    raise ValueError('VLM did not capture all section breaks')
                
            # ---------------------------------------------------------------------------------------
            # STEP 6A: Define coordinates of panels horseshoe vortices and control points 
            # --------------------------------------------------------------------------------------- 
            y_a   = y_coordinates[:-1] 
            y_b   = y_coordinates[1:]             
            del_y = y_coordinates[1:] - y_coordinates[:-1]           
            i_seg = 0           
            for idx_y in range(n_sw):
                # define coordinates of horseshoe vortices and control points
                idx_x = np.arange(n_cw) 
                eta_a = (y_a[idx_y] - section_stations[i_seg])  
                eta_b = (y_b[idx_y] - section_stations[i_seg]) 
                eta   = (y_b[idx_y] - del_y[idx_y]/2 - section_stations[i_seg]) 

                segment_chord_ratio = (segment_chord[i_seg+1] - segment_chord[i_seg])/segment_span[i_seg+1]
                segment_twist_ratio = (segment_twist[i_seg+1] - segment_twist[i_seg])/segment_span[i_seg+1]

                wing_chord_section_a  = segment_chord[i_seg] + (eta_a*segment_chord_ratio) 
                wing_chord_section_b  = segment_chord[i_seg] + (eta_b*segment_chord_ratio)
                wing_chord_section    = segment_chord[i_seg] + (eta*segment_chord_ratio)

                delta_x_a = wing_chord_section_a/n_cw  
                delta_x_b = wing_chord_section_b/n_cw      
                delta_x   = wing_chord_section/n_cw                                       

                xi_a1 = segment_chord_x_offset[i_seg] + eta_a*np.tan(segment_sweep[i_seg]) + delta_x_a*idx_x                  # x coordinate of top left corner of panel 
                xi_a2 = segment_chord_x_offset[i_seg] + eta_a*np.tan(segment_sweep[i_seg]) + delta_x_a*idx_x + delta_x_a      # x coordinate of bottom left corner of bound vortex  
                xi_b1 = segment_chord_x_offset[i_seg] + eta_b*np.tan(segment_sweep[i_seg]) + delta_x_b*idx_x                  # x coordinate of top right corner of panel     
                xi_b2 = segment_chord_x_offset[i_seg] + eta_b*np.tan(segment_sweep[i_seg]) + delta_x_b*idx_x + delta_x_b      # x coordinate of bottom right corner of panel      
                xi_c  = segment_chord_x_offset[i_seg] + eta *np.tan(segment_sweep[i_seg])  + delta_x  *idx_x + delta_x*0.75   # x coordinate three-quarter chord control point for each panel
                 

                # adjustment of coordinates for camber
                section_camber_a  = segment_camber[i_seg]*wing_chord_section_a  
                section_top_a     = segment_top_surface[i_seg]*wing_chord_section_a
                section_bot_a     = segment_bot_surface[i_seg]*wing_chord_section_a
                section_camber_b  = segment_camber[i_seg]*wing_chord_section_b  
                section_top_b     = segment_top_surface[i_seg]*wing_chord_section_b
                section_bot_b     = segment_bot_surface[i_seg]*wing_chord_section_b                
                section_camber_c  = segment_camber[i_seg]*wing_chord_section                
                section_x_coord_a = segment_x_coord[i_seg]*wing_chord_section_a
                section_x_coord_b = segment_x_coord[i_seg]*wing_chord_section_b
                section_x_coord   = segment_x_coord[i_seg]*wing_chord_section

                z_c_a1     = np.interp((idx_x    *delta_x_a)                  ,section_x_coord_a,section_camber_a)  
                z_c_a1_top = np.interp((idx_x    *delta_x_a)                  ,section_x_coord_a,section_top_a)
                z_c_a1_bot = np.interp((idx_x    *delta_x_a)                  ,section_x_coord_a,section_bot_a) 
                z_c_a2     = np.interp(((idx_x+1)*delta_x_a)                  ,section_x_coord_a,section_camber_a)  
                z_c_a2_top = np.interp(((idx_x+1)*delta_x_a)                  ,section_x_coord_a,section_top_a) 
                z_c_a2_bot = np.interp(((idx_x+1)*delta_x_a)                  ,section_x_coord_a,section_bot_a)   
                z_c_b1     = np.interp((idx_x    *delta_x_b)                  ,section_x_coord_b,section_camber_b)  
                z_c_b1_top = np.interp((idx_x    *delta_x_b)                  ,section_x_coord_b,section_top_b)  
                z_c_b1_bot = np.interp((idx_x    *delta_x_b)                  ,section_x_coord_b,section_bot_b)  
                z_c_b2     = np.interp(((idx_x+1)*delta_x_b)                  ,section_x_coord_b,section_camber_b) 
                z_c_b2_top = np.interp(((idx_x+1)*delta_x_b)                  ,section_x_coord_b,section_top_b) 
                z_c_b2_bot = np.interp(((idx_x+1)*delta_x_b)                  ,section_x_coord_b,section_bot_b)   
                z_c        = np.interp((idx_x    *delta_x   + delta_x  *0.75) ,section_x_coord,section_camber_c)  

                zeta_a1     = segment_chord_z_offset[i_seg] + eta_a*np.tan(segment_dihedral[i_seg])  + z_c_a1      # z coordinate of top left corner of panel 
                zeta_a1_top = segment_chord_z_offset[i_seg] + eta_a*np.tan(segment_dihedral[i_seg])  + z_c_a1_top
                zeta_a1_bot = segment_chord_z_offset[i_seg] + eta_a*np.tan(segment_dihedral[i_seg])  + z_c_a1_bot 
                zeta_a2     = segment_chord_z_offset[i_seg] + eta_a*np.tan(segment_dihedral[i_seg])  + z_c_a2      # z coordinate of bottom left corner of panel
                zeta_a2_top = segment_chord_z_offset[i_seg] + eta_a*np.tan(segment_dihedral[i_seg])  + z_c_a2_top 
                zeta_a2_bot = segment_chord_z_offset[i_seg] + eta_a*np.tan(segment_dihedral[i_seg])  + z_c_a2_bot                         
                zeta_b1     = segment_chord_z_offset[i_seg] + eta_b*np.tan(segment_dihedral[i_seg])  + z_c_b1      # z coordinate of top right corner of panel  
                zeta_b1_top = segment_chord_z_offset[i_seg] + eta_b*np.tan(segment_dihedral[i_seg])  + z_c_b1_top
                zeta_b1_bot = segment_chord_z_offset[i_seg] + eta_b*np.tan(segment_dihedral[i_seg])  + z_c_b1_bot   
                zeta_b2     = segment_chord_z_offset[i_seg] + eta_b*np.tan(segment_dihedral[i_seg])  + z_c_b2      # z coordinate of bottom right corner of panel        
                zeta_b2_top = segment_chord_z_offset[i_seg] + eta_b*np.tan(segment_dihedral[i_seg])  + z_c_b2_top 
                zeta_b2_bot = segment_chord_z_offset[i_seg] + eta_b*np.tan(segment_dihedral[i_seg])  + z_c_b2_bot 
                zeta        = segment_chord_z_offset[i_seg] + eta*np.tan(segment_dihedral[i_seg])    + z_c         # z coordinate three-quarter chord control point for each panel 

                # adjustment of coordinates for twist  
                xi_LE_a = segment_chord_x_offset[i_seg] + eta_a*np.tan(segment_sweep[i_seg])                       # x location of leading edge left corner of wing
                xi_LE_b = segment_chord_x_offset[i_seg] + eta_b*np.tan(segment_sweep[i_seg])                       # x location of leading edge right of wing
                xi_LE   = segment_chord_x_offset[i_seg] + eta*np.tan(segment_sweep[i_seg])                         # x location of leading edge center of wing
                                                                                                                   
                zeta_LE_a = segment_chord_z_offset[i_seg] + eta_a*np.tan(segment_dihedral[i_seg])                  # z location of leading edge left corner of wing
                zeta_LE_b = segment_chord_z_offset[i_seg] + eta_b*np.tan(segment_dihedral[i_seg])                  # z location of leading edge right of wing
                zeta_LE   = segment_chord_z_offset[i_seg] + eta*np.tan(segment_dihedral[i_seg])                    # z location of leading edge center of wing
                                                                                                                   
                # determine section twist                                                                          
                section_twist_a = segment_twist[i_seg] + (eta_a * segment_twist_ratio)                             # twist at left side of panel
                section_twist_b = segment_twist[i_seg] + (eta_b * segment_twist_ratio)                             # twist at right side of panel
                section_twist   = segment_twist[i_seg] + (eta* segment_twist_ratio)                                # twist at center local chord 

                xi_prime_a1        = xi_LE_a + np.cos(section_twist_a)*(xi_a1-xi_LE_a) + np.sin(section_twist_a)*(zeta_a1-zeta_LE_a)        # x coordinate transformation of top left corner
                xi_prime_a1_top    = xi_LE_a + np.cos(section_twist_a)*(xi_a1-xi_LE_a) + np.sin(section_twist_a)*(zeta_a1_top-zeta_LE_a)    
                xi_prime_a1_bot    = xi_LE_a + np.cos(section_twist_a)*(xi_a1-xi_LE_a) + np.sin(section_twist_a)*(zeta_a1_bot-zeta_LE_a)    
                xi_prime_a2        = xi_LE_a + np.cos(section_twist_a)*(xi_a2-xi_LE_a) + np.sin(section_twist_a)*(zeta_a2-zeta_LE_a)        # x coordinate transformation of bottom left corner
                xi_prime_a2_top    = xi_LE_a + np.cos(section_twist_a)*(xi_a2-xi_LE_a) + np.sin(section_twist_a)*(zeta_a2_top-zeta_LE_a)    
                xi_prime_a2_bot    = xi_LE_a + np.cos(section_twist_a)*(xi_a2-xi_LE_a) + np.sin(section_twist_a)*(zeta_a2_bot-zeta_LE_a)                          
                xi_prime_b1        = xi_LE_b + np.cos(section_twist_b)*(xi_b1-xi_LE_b) + np.sin(section_twist_b)*(zeta_b1-zeta_LE_b)        # x coordinate transformation of top right corner  
                xi_prime_b1_top    = xi_LE_b + np.cos(section_twist_b)*(xi_b1-xi_LE_b) + np.sin(section_twist_b)*(zeta_b1_top-zeta_LE_b)    
                xi_prime_b1_bot    = xi_LE_b + np.cos(section_twist_b)*(xi_b1-xi_LE_b) + np.sin(section_twist_b)*(zeta_b1_bot-zeta_LE_b)    
                xi_prime_b2        = xi_LE_b + np.cos(section_twist_b)*(xi_b2-xi_LE_b) + np.sin(section_twist_b)*(zeta_b2-zeta_LE_b)        # x coordinate transformation of botton right corner 
                xi_prime_b2_top    = xi_LE_b + np.cos(section_twist_b)*(xi_b2-xi_LE_b) + np.sin(section_twist_b)*(zeta_b2_top-zeta_LE_b)    
                xi_prime_b2_bot    = xi_LE_b + np.cos(section_twist_b)*(xi_b2-xi_LE_b) + np.sin(section_twist_b)*(zeta_b2_bot-zeta_LE_b)    
                xi_prime           = xi_LE   + np.cos(section_twist)  *(xi_c-xi_LE)    + np.sin(section_twist)*(zeta-zeta_LE)               # x coordinate transformation of control point 

                zeta_prime_a1      = zeta_LE_a - np.sin(section_twist_a)*(xi_a1-xi_LE_a) + np.cos(section_twist_a)*(zeta_a1-zeta_LE_a)      # z coordinate transformation of top left corner 
                zeta_prime_a1_top  = zeta_LE_a - np.sin(section_twist_a)*(xi_a1-xi_LE_a) + np.cos(section_twist_a)*(zeta_a1_top-zeta_LE_a)
                zeta_prime_a1_bot  = zeta_LE_a - np.sin(section_twist_a)*(xi_a1-xi_LE_a) + np.cos(section_twist_a)*(zeta_a1_bot-zeta_LE_a)
                zeta_prime_a2      = zeta_LE_a - np.sin(section_twist_a)*(xi_a2-xi_LE_a) + np.cos(section_twist_a)*(zeta_a2-zeta_LE_a)      # z coordinate transformation of bottom left corner
                zeta_prime_a2_top  = zeta_LE_a - np.sin(section_twist_a)*(xi_a2-xi_LE_a) + np.cos(section_twist_a)*(zeta_a2_top-zeta_LE_a)
                zeta_prime_a2_bot  = zeta_LE_a - np.sin(section_twist_a)*(xi_a2-xi_LE_a) + np.cos(section_twist_a)*(zeta_a2_bot-zeta_LE_a)                         
                zeta_prime_b1      = zeta_LE_b - np.sin(section_twist_b)*(xi_b1-xi_LE_b) + np.cos(section_twist_b)*(zeta_b1-zeta_LE_b)      # z coordinate transformation of top right corner  
                zeta_prime_b1_top  = zeta_LE_b - np.sin(section_twist_b)*(xi_b1-xi_LE_b) + np.cos(section_twist_b)*(zeta_b1_top-zeta_LE_b)
                zeta_prime_b1_bot  = zeta_LE_b - np.sin(section_twist_b)*(xi_b1-xi_LE_b) + np.cos(section_twist_b)*(zeta_b1_bot-zeta_LE_b)
                zeta_prime_b2      = zeta_LE_b - np.sin(section_twist_b)*(xi_b2-xi_LE_b) + np.cos(section_twist_b)*(zeta_b2-zeta_LE_b)      # z coordinate transformation of botton right corner 
                zeta_prime_b2_top  = zeta_LE_b - np.sin(section_twist_b)*(xi_b2-xi_LE_b) + np.cos(section_twist_b)*(zeta_b2_top-zeta_LE_b) 
                zeta_prime_b2_bot  = zeta_LE_b - np.sin(section_twist_b)*(xi_b2-xi_LE_b) + np.cos(section_twist_b)*(zeta_b2_bot-zeta_LE_b) 
                zeta_prime         = zeta_LE   - np.sin(section_twist)*(xi_c-xi_LE)      + np.cos(-section_twist)*(zeta-zeta_LE)            # z coordinate transformation of control point 

                # ** TO DO ** Get flap/aileron locations and deflection
                # store coordinates of panels, horseshoeces vortices and control points relative to wing root 
                if vertical_wing:
                    # mean camber line surface 
                    xa1[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_a1 
                    za1[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                    ya1[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_a1
                    xa2[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_a2
                    za2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                    ya2[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_a2 
                    xb1[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_b1 
                    zb1[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                    yb1[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_b1
                    xb2[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_b2 
                    zb2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]                        
                    yb2[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_b2  
                                                                                                                                   
                    xa1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_a1_top          ,xi_prime_a1_bot              ])
                    za1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_a[idx_y] ,np.ones(n_cw)*y_a[idx_y]     ])
                    ya1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_a1_top        ,zeta_prime_a1_bot            ])
                    xa2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_a2_top          ,xi_prime_a2_bot              ])
                    za2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_a[idx_y] ,np.ones(n_cw)*y_a[idx_y]     ])
                    ya2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_a2_top        ,zeta_prime_a2_bot            ])
                    xb1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_b1_top          ,xi_prime_b1_bot              ])
                    zb1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_b[idx_y] ,np.ones(n_cw)*y_b[idx_y]     ])
                    yb1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_b1_top        ,zeta_prime_b1_bot            ])
                    xb2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_b2_top          ,xi_prime_b2_bot              ])
                    zb2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_b[idx_y] ,np.ones(n_cw)*y_b[idx_y]     ])                      
                    yb2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_b2_top        ,zeta_prime_b2_bot            ]) 

                    xc[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime 
                    zc[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*(y_b[idx_y] - del_y[idx_y]/2) 
                    yc[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime 

                else:     
                    xa1[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_a1 
                    ya1[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                    za1[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_a1
                    xa2[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_a2
                    ya2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                    za2[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_a2 
                    xb1[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_b1 
                    yb1[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                    zb1[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_b1
                    xb2[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_b2
                    yb2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                    zb2[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_b2 
                    
                    xa1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_a1_top          ,xi_prime_a1_bot              ])
                    ya1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_a[idx_y] ,np.ones(n_cw)*y_a[idx_y]     ])
                    za1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_a1_top        ,zeta_prime_a1_bot            ])
                    xa2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_a2_top          ,xi_prime_a2_bot              ])
                    ya2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_a[idx_y] ,np.ones(n_cw)*y_a[idx_y]     ])
                    za2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_a2_top        ,zeta_prime_a2_bot            ])
                    xb1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_b1_top          ,xi_prime_b1_bot              ])
                    yb1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_b[idx_y] ,np.ones(n_cw)*y_b[idx_y]     ])
                    zb1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_b1_top        ,zeta_prime_b1_bot            ])
                    xb2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_b2_top          ,xi_prime_b2_bot              ])
                    yb2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_b[idx_y] ,np.ones(n_cw)*y_b[idx_y]     ])                      
                    zb2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_b2_top        ,zeta_prime_b2_bot            ])
 
                    xc [idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime 
                    yc [idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*(y_b[idx_y] - del_y[idx_y]/2)
                    zc [idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime                  

                idx += 1

                cs_w[idx_y] = wing_chord_section       

                if y_b[idx_y] == section_stations[i_seg+1]: 
                    i_seg += 1                     

        else:   # when no segments are defined on wing  
            # ---------------------------------------------------------------------------------------
            # STEP 6B: Define coordinates of panels horseshoe vortices and control points 
            # ---------------------------------------------------------------------------------------
            y_a   = y_coordinates[:-1] 
            y_b   = y_coordinates[1:] 
            
            if sweep_le != None:
                sweep = sweep_le
            else:                                                                
                cf    = 0.25                          
                sweep = np.arctan(((root_chord*cf) + (np.tan(sweep_qc)*span - cf*tip_chord)) /span)  
           
            wing_chord_ratio = (tip_chord-root_chord)/span
            wing_twist_ratio = (twist_tc-twist_rc)/span                    
            wing_areas.append(0.5*(root_chord+tip_chord)*span) 
            if sym_para is True :
                wing_areas.append(0.5*(root_chord+tip_chord)*span)   

            # Get airfoil section VD  
            if wing.Airfoil: 
                airfoil_data     = import_airfoil_geometry([wing.Airfoil.airfoil.coordinate_file])    
                wing_camber      = airfoil_data.camber_coordinates[0]
                wing_top_surface = airfoil_data.y_upper_surface[0] 
                wing_bot_surface = airfoil_data.y_lower_surface[0] 
                wing_x_coord     = airfoil_data.x_lower_surface[0]
            else:
                dummy_dimension  = 30
                wing_camber      = np.zeros(dummy_dimension) # dimension of Selig airfoil VD file
                airfoil_data     = compute_naca_4series(0.0, 0.0,wing.thickness_to_chord,dummy_dimension*2 - 2 )
                wing_top_surface = airfoil_data.y_upper_surface[0] 
                wing_bot_surface = airfoil_data.y_lower_surface[0]                 
                wing_x_coord     = np.linspace(0,1,30) 
                    
            del_y = y_b - y_a
            for idx_y in range(n_sw):  
                idx_x = np.arange(n_cw) 
                eta_a = (y_a[idx_y])  
                eta_b = (y_b[idx_y]) 
                eta   = (y_b[idx_y] - del_y[idx_y]/2) 
                
                # get spanwise discretization points
                wing_chord_section_a  = root_chord + (eta_a*wing_chord_ratio) 
                wing_chord_section_b  = root_chord + (eta_b*wing_chord_ratio)
                wing_chord_section    = root_chord + (eta*wing_chord_ratio)
                
                # get chordwise discretization points
                delta_x_a = wing_chord_section_a/n_cw   
                delta_x_b = wing_chord_section_b/n_cw   
                delta_x   = wing_chord_section/n_cw                                  

                xi_a1 = eta_a*np.tan(sweep) + delta_x_a*idx_x                  # x coordinate of top left corner of panel 
                xi_a2 = eta_a*np.tan(sweep) + delta_x_a*idx_x + delta_x_a      # x coordinate of bottom left corner of bound vortex  
                xi_b1 = eta_b*np.tan(sweep) + delta_x_b*idx_x                  # x coordinate of top right corner of panel            
                xi_b2 = eta_b*np.tan(sweep) + delta_x_b*idx_x + delta_x_b
                xi_c  =  eta *np.tan(sweep)  + delta_x  *idx_x + delta_x*0.75  # x coordinate three-quarter chord control point for each panel 
                
                # adjustment of coordinates for camber
                section_camber_a  = wing_camber*wing_chord_section_a
                section_top_a     = wing_top_surface*wing_chord_section_a
                section_bot_a     = wing_bot_surface*wing_chord_section_a
                section_camber_b  = wing_camber*wing_chord_section_b                  
                section_top_b     = wing_top_surface*wing_chord_section_b                  
                section_bot_b     = wing_bot_surface*wing_chord_section_b  
                section_camber_c  = wing_camber*wing_chord_section  

                section_x_coord_a = wing_x_coord*wing_chord_section_a
                section_x_coord_b = wing_x_coord*wing_chord_section_b
                section_x_coord   = wing_x_coord*wing_chord_section 

                z_c_a1     = np.interp((idx_x    *delta_x_a)                  ,section_x_coord_a,section_camber_a)  
                z_c_a1_top = np.interp((idx_x    *delta_x_a)                  ,section_x_coord_a,section_top_a) 
                z_c_a1_bot = np.interp((idx_x    *delta_x_a)                  ,section_x_coord_a,section_bot_a) 
                z_c_a2     = np.interp(((idx_x+1)*delta_x_a)                  ,section_x_coord_a,section_camber_a) 
                z_c_a2_top = np.interp(((idx_x+1)*delta_x_a)                  ,section_x_coord_a,section_top_a)
                z_c_a2_bot = np.interp(((idx_x+1)*delta_x_a)                  ,section_x_coord_a,section_bot_a) 
                z_c_b1     = np.interp((idx_x    *delta_x_b)                  ,section_x_coord_b,section_camber_b)    
                z_c_b1_top = np.interp((idx_x    *delta_x_b)                  ,section_x_coord_b,section_top_b)
                z_c_b1_bot = np.interp((idx_x    *delta_x_b)                  ,section_x_coord_b,section_bot_b)
                z_c_b2     = np.interp(((idx_x+1)*delta_x_b)                  ,section_x_coord_b,section_camber_b) 
                z_c_b2_top = np.interp(((idx_x+1)*delta_x_b)                  ,section_x_coord_b,section_top_b) 
                z_c_b2_bot = np.interp(((idx_x+1)*delta_x_b)                  ,section_x_coord_b,section_bot_b)  
                z_c        = np.interp((idx_x    *delta_x   + delta_x  *0.75) ,section_x_coord  ,section_camber_c)  

                zeta_a1     = eta_a*np.tan(dihedral)  + z_c_a1      # z coordinate of top left corner of panel 
                zeta_a1_top = eta_a*np.tan(dihedral)  + z_c_a1_top  # z coordinate of top left corner of panel on surface  
                zeta_a1_bot = eta_a*np.tan(dihedral)  + z_c_a1_bot  # z coordinate of top left corner of panel on surface  
                zeta_a2     = eta_a*np.tan(dihedral)  + z_c_a2      # z coordinate of bottom left corner of panel
                zeta_a2_top = eta_a*np.tan(dihedral)  + z_c_a2_top  # z coordinate of bottom left corner of panel
                zeta_a2_bot = eta_a*np.tan(dihedral)  + z_c_a2_bot  # z coordinate of bottom left corner of panel                    
                zeta_b1     = eta_b*np.tan(dihedral)  + z_c_b1      # z coordinate of top right corner of panel    
                zeta_b1_top = eta_b*np.tan(dihedral)  + z_c_b1_top  # z coordinate of top right corner of panel    
                zeta_b1_bot = eta_b*np.tan(dihedral)  + z_c_b1_bot  # z coordinate of top right corner of panel    
                zeta_b2     = eta_b*np.tan(dihedral)  + z_c_b2      # z coordinate of bottom right corner of panel   
                zeta_b2_top = eta_b*np.tan(dihedral)  + z_c_b2_top  # z coordinate of bottom right corner of panel  
                zeta_b2_bot = eta_b*np.tan(dihedral)  + z_c_b2_bot  # z coordinate of bottom right corner of panel  
                zeta        =   eta*np.tan(dihedral)  + z_c         # z coordinate three-quarter chord control point for each panel  

                # adjustment of coordinates for twist  
                xi_LE_a = eta_a*np.tan(sweep)               # x location of leading edge left corner of wing
                xi_LE_b = eta_b*np.tan(sweep)               # x location of leading edge right of wing
                xi_LE   = eta  *np.tan(sweep)               # x location of leading edge center of wing

                zeta_LE_a = eta_a*np.tan(dihedral)          # z location of leading edge left corner of wing
                zeta_LE_b = eta_b*np.tan(dihedral)          # z location of leading edge right of wing
                zeta_LE   = eta  *np.tan(dihedral)          # z location of leading edge center of wing

                # determine section twist
                section_twist_a = twist_rc + (eta_a * wing_twist_ratio)                     # twist at left side of panel
                section_twist_b = twist_rc + (eta_b * wing_twist_ratio)                     # twist at right side of panel
                section_twist   = twist_rc + (eta   * wing_twist_ratio)                     # twist at center local chord 
 
                xi_prime_a1      = xi_LE_a + np.cos(section_twist_a)*(xi_a1-xi_LE_a) + np.sin(section_twist_a)*(zeta_a1-zeta_LE_a)         # x coordinate transformation of top left corner
                xi_prime_a1_top  = xi_LE_a + np.cos(section_twist_a)*(xi_a1-xi_LE_a) + np.sin(section_twist_a)*(zeta_a1_top-zeta_LE_a)     
                xi_prime_a1_bot  = xi_LE_a + np.cos(section_twist_a)*(xi_a1-xi_LE_a) + np.sin(section_twist_a)*(zeta_a1_bot-zeta_LE_a)      
                xi_prime_a2      = xi_LE_a + np.cos(section_twist_a)*(xi_a2-xi_LE_a) + np.sin(section_twist_a)*(zeta_a2-zeta_LE_a)         # x coordinate transformation of bottom left corner
                xi_prime_a2_top  = xi_LE_a + np.cos(section_twist_a)*(xi_a2-xi_LE_a) + np.sin(section_twist_a)*(zeta_a2_top-zeta_LE_a)      
                xi_prime_a2_bot  = xi_LE_a + np.cos(section_twist_a)*(xi_a2-xi_LE_a) + np.sin(section_twist_a)*(zeta_a2_bot-zeta_LE_a)                             
                xi_prime_b1      = xi_LE_b + np.cos(section_twist_b)*(xi_b1-xi_LE_b) + np.sin(section_twist_b)*(zeta_b1-zeta_LE_b)         # x coordinate transformation of top right corner  
                xi_prime_b1_top  = xi_LE_b + np.cos(section_twist_b)*(xi_b1-xi_LE_b) + np.sin(section_twist_b)*(zeta_b1_top-zeta_LE_b)      
                xi_prime_b1_bot  = xi_LE_b + np.cos(section_twist_b)*(xi_b1-xi_LE_b) + np.sin(section_twist_b)*(zeta_b1_bot-zeta_LE_b)     
                xi_prime_b2      = xi_LE_b + np.cos(section_twist_b)*(xi_b2-xi_LE_b) + np.sin(section_twist_b)*(zeta_b2-zeta_LE_b)         # x coordinate transformation of botton right corner 
                xi_prime_b2_top  = xi_LE_b + np.cos(section_twist_b)*(xi_b2-xi_LE_b) + np.sin(section_twist_b)*(zeta_b2_top-zeta_LE_b)    
                xi_prime_b2_bot  = xi_LE_b + np.cos(section_twist_b)*(xi_b2-xi_LE_b) + np.sin(section_twist_b)*(zeta_b2_bot-zeta_LE_b)    
                xi_prime         = xi_LE   + np.cos(section_twist)  *(xi_c-xi_LE)    + np.sin(section_twist)*(zeta-zeta_LE)                # x coordinate transformation of control point 

                zeta_prime_a1      = zeta_LE_a - np.sin(section_twist_a)*(xi_a1-xi_LE_a) + np.cos(section_twist_a)*(zeta_a1-zeta_LE_a)     # z coordinate transformation of top left corner 
                zeta_prime_a1_top  = zeta_LE_a - np.sin(section_twist_a)*(xi_a1-xi_LE_a) + np.cos(section_twist_a)*(zeta_a1_top-zeta_LE_a)
                zeta_prime_a1_bot  = zeta_LE_a - np.sin(section_twist_a)*(xi_a1-xi_LE_a) + np.cos(section_twist_a)*(zeta_a1_bot-zeta_LE_a)
                zeta_prime_a2      = zeta_LE_a - np.sin(section_twist_a)*(xi_a2-xi_LE_a) + np.cos(section_twist_a)*(zeta_a2-zeta_LE_a)     # z coordinate transformation of bottom left corner
                zeta_prime_a2_top  = zeta_LE_a - np.sin(section_twist_a)*(xi_a2-xi_LE_a) + np.cos(section_twist_a)*(zeta_a2_top-zeta_LE_a)
                zeta_prime_a2_bot  = zeta_LE_a - np.sin(section_twist_a)*(xi_a2-xi_LE_a) + np.cos(section_twist_a)*(zeta_a2_bot-zeta_LE_a)                         
                zeta_prime_b1      = zeta_LE_b - np.sin(section_twist_b)*(xi_b1-xi_LE_b) + np.cos(section_twist_b)*(zeta_b1-zeta_LE_b)     # z coordinate transformation of top right corner  
                zeta_prime_b1_top  = zeta_LE_b - np.sin(section_twist_b)*(xi_b1-xi_LE_b) + np.cos(section_twist_b)*(zeta_b1_top-zeta_LE_b)
                zeta_prime_b1_bot  = zeta_LE_b - np.sin(section_twist_b)*(xi_b1-xi_LE_b) + np.cos(section_twist_b)*(zeta_b1_bot-zeta_LE_b)
                zeta_prime_b2      = zeta_LE_b - np.sin(section_twist_b)*(xi_b2-xi_LE_b) + np.cos(section_twist_b)*(zeta_b2-zeta_LE_b)      # z coordinate transformation of botton right corner 
                zeta_prime_b2_top  = zeta_LE_b - np.sin(section_twist_b)*(xi_b2-xi_LE_b) + np.cos(section_twist_b)*(zeta_b2_top-zeta_LE_b) 
                zeta_prime_b2_bot  = zeta_LE_b - np.sin(section_twist_b)*(xi_b2-xi_LE_b) + np.cos(section_twist_b)*(zeta_b2_bot-zeta_LE_b) 
                zeta_prime         = zeta_LE   - np.sin(section_twist)*(xi_c-xi_LE)      + np.cos(-section_twist)*(zeta-zeta_LE)            # z coordinate transformation of control point 
  
 
                # store coordinates of panels, horseshoe vortices and control points relative to wing root 
                if vertical_wing: 
                    
                    # mean camber line surface 
                    xa1[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_a1 
                    za1[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                    ya1[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_a1
                    xa2[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_a2
                    za2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                    ya2[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_a2 
                    xb1[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_b1 
                    zb1[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                    yb1[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_b1
                    xb2[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_b2 
                    zb2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]                        
                    yb2[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_b2  
                
                    xa1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_a1_top          ,xi_prime_a1_bot              ])
                    za1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_a[idx_y] ,np.ones(n_cw)*y_a[idx_y]     ])
                    ya1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_a1_top        ,zeta_prime_a1_bot            ])
                    xa2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_a2_top          ,xi_prime_a2_bot              ])
                    za2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_a[idx_y] ,np.ones(n_cw)*y_a[idx_y]     ])
                    ya2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_a2_top        ,zeta_prime_a2_bot            ])
                    xb1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_b1_top          ,xi_prime_b1_bot              ])
                    zb1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_b[idx_y] ,np.ones(n_cw)*y_b[idx_y]     ])
                    yb1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_b1_top        ,zeta_prime_b1_bot            ])
                    xb2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_b2_top          ,xi_prime_b2_bot              ])
                    zb2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_b[idx_y] ,np.ones(n_cw)*y_b[idx_y]     ])                      
                    yb2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_b2_top        ,zeta_prime_b2_bot            ])   

                    xc [idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime 
                    zc [idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*(y_b[idx_y] - del_y[idx_y]/2) 
                    yc [idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime                       

                else: 
                    xa1[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_a1 
                    ya1[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                    za1[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_a1
                    xa2[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_a2
                    ya2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                    za2[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_a2 
                    xb1[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_b1 
                    yb1[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                    zb1[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_b1
                    xb2[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_b2
                    yb2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                    zb2[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_b2   
                    
                    xa1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_a1_top          ,xi_prime_a1_bot              ])
                    ya1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_a[idx_y] ,np.ones(n_cw)*y_a[idx_y]     ])
                    za1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_a1_top        ,zeta_prime_a1_bot            ])
                    xa2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_a2_top          ,xi_prime_a2_bot              ])
                    ya2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_a[idx_y] ,np.ones(n_cw)*y_a[idx_y]     ])
                    za2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_a2_top        ,zeta_prime_a2_bot            ])
                    xb1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_b1_top          ,xi_prime_b1_bot              ])
                    yb1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_b[idx_y] ,np.ones(n_cw)*y_b[idx_y]     ])
                    zb1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_b1_top        ,zeta_prime_b1_bot            ])
                    xb2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_b2_top          ,xi_prime_b2_bot              ])
                    yb2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_b[idx_y] ,np.ones(n_cw)*y_b[idx_y]     ])                      
                    zb2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_b2_top        ,zeta_prime_b2_bot            ])
                    
                    
                    xc [idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime 
                    yc [idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*(y_b[idx_y] - del_y[idx_y]/2) 
                    zc [idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime       

                cs_w[idx_y] = wing_chord_section 
    
        # adjusting coordinate axis so reference point is at the nose of the aircraft 

        xa1 = xa1 + wing_origin[0]         # x coordinate of top left corner of panel
        ya1 = ya1 + wing_origin[1]         # y coordinate of bottom left corner of panel
        za1 = za1 + wing_origin[2]         # z coordinate of top left corner of panel
        xa2 = xa2 + wing_origin[0]         # x coordinate of bottom left corner of panel
        ya2 = ya2 + wing_origin[1]         # x coordinate of bottom left corner of panel
        za2 = za2 + wing_origin[2]         # z coordinate of bottom left corner of panel   
        xb1 = xb1 + wing_origin[0]         # x coordinate of top right corner of panel  
        yb1 = yb1 + wing_origin[1]         # y coordinate of top right corner of panel 
        zb1 = zb1 + wing_origin[2]         # z coordinate of top right corner of panel 
        xb2 = xb2 + wing_origin[0]         # x coordinate of bottom rightcorner of panel 
        yb2 = yb2 + wing_origin[1]         # y coordinate of bottom rightcorner of panel 
        zb2 = zb2 + wing_origin[2]         # z coordinate of bottom right corner of panel            
        
        xa1_ts   = xa1_ts + wing_origin[0] # x coordinate of top left corner of panel
        ya1_ts   = ya1_ts + wing_origin[1] # y coordinate of bottom left corner of panel
        za1_ts   = za1_ts + wing_origin[2] # z coordinate of top left corner of panel
        xa2_ts   = xa2_ts + wing_origin[0] # x coordinate of bottom left corner of panel
        ya2_ts   = ya2_ts + wing_origin[1] # x coordinate of bottom left corner of panel
        za2_ts   = za2_ts + wing_origin[2] # z coordinate of bottom left corner of panel   
        xb1_ts   = xb1_ts + wing_origin[0] # x coordinate of top right corner of panel  
        yb1_ts   = yb1_ts + wing_origin[1] # y coordinate of top right corner of panel 
        zb1_ts   = zb1_ts + wing_origin[2] # z coordinate of top right corner of panel 
        xb2_ts   = xb2_ts + wing_origin[0] # x coordinate of bottom rightcorner of panel 
        yb2_ts   = yb2_ts + wing_origin[1] # y coordinate of bottom rightcorner of panel 
        zb2_ts   = zb2_ts + wing_origin[2] # z coordinate of bottom right corner of panel      

        # if symmetry, store points of mirrored wing 
        n_w += 1  
        if sym_para is True :
            n_w += 1 
            # append wing spans          
            if vertical_wing:
                del_y    = np.concatenate([del_y,del_y]) 
                cs_w     = np.concatenate([cs_w,cs_w]) 
                         
                xa1   = np.concatenate([xa1,xa1])
                ya1   = np.concatenate([ya1,ya1])
                za1   = np.concatenate([za1,-za1])
                xa2   = np.concatenate([xa2,xa2])
                ya2   = np.concatenate([ya2,ya2])
                za2   = np.concatenate([za2,-za2]) 
                xb1   = np.concatenate([xb1,xb1])
                yb1   = np.concatenate([yb1,yb1])    
                zb1   = np.concatenate([zb1,-zb1])
                xb2   = np.concatenate([xb2,xb2])
                yb2   = np.concatenate([yb2,yb2])            
                zb2   = np.concatenate([zb2,-zb2])
                
                xa1_ts   = np.concatenate([xa1_ts, xa1_ts])
                ya1_ts   = np.concatenate([ya1_ts, ya1_ts])
                za1_ts   = np.concatenate([za1_ts,-za1_ts])
                xa2_ts   = np.concatenate([xa2_ts, xa2_ts])
                ya2_ts   = np.concatenate([ya2_ts, ya2_ts])
                za2_ts   = np.concatenate([za2_ts,-za2_ts]) 
                xb1_ts   = np.concatenate([xb1_ts, xb1_ts])
                yb1_ts   = np.concatenate([yb1_ts, yb1_ts])    
                zb1_ts   = np.concatenate([zb1_ts,-zb1_ts])
                xb2_ts   = np.concatenate([xb2_ts, xb2_ts])
                yb2_ts   = np.concatenate([yb2_ts, yb2_ts])            
                zb2_ts   = np.concatenate([zb2_ts,-zb2_ts])
                 
                xc       = np.concatenate([xc ,xc ])
                yc       = np.concatenate([yc ,yc]) 
                zc       = np.concatenate([zc ,-zc ])                 
                
            else: 
                xa1   = np.concatenate([xa1,xa1])
                ya1   = np.concatenate([ya1,-ya1])
                za1   = np.concatenate([za1,za1])
                xa2   = np.concatenate([xa2,xa2])
                ya2   = np.concatenate([ya2,-ya2])
                za2   = np.concatenate([za2,za2]) 
                xb1   = np.concatenate([xb1,xb1])
                yb1   = np.concatenate([yb1,-yb1])    
                zb1   = np.concatenate([zb1,zb1])
                xb2   = np.concatenate([xb2,xb2])
                yb2   = np.concatenate([yb2,-yb2])            
                zb2   = np.concatenate([zb2,zb2])
                
                xa1_ts  = np.concatenate([xa1_ts, xa1_ts])
                ya1_ts  = np.concatenate([ya1_ts,-ya1_ts])
                za1_ts  = np.concatenate([za1_ts, za1_ts])
                xa2_ts  = np.concatenate([xa2_ts, xa2_ts])
                ya2_ts  = np.concatenate([ya2_ts,-ya2_ts])
                za2_ts  = np.concatenate([za2_ts, za2_ts]) 
                xb1_ts  = np.concatenate([xb1_ts, xb1_ts])
                yb1_ts  = np.concatenate([yb1_ts,-yb1_ts])    
                zb1_ts  = np.concatenate([zb1_ts, zb1_ts])
                xb2_ts  = np.concatenate([xb2_ts, xb2_ts])
                yb2_ts  = np.concatenate([yb2_ts,-yb2_ts])            
                zb2_ts  = np.concatenate([zb2_ts, zb2_ts])
                
                xc       = np.concatenate([xc , xc ])
                yc       = np.concatenate([yc ,-yc]) 
                zc       = np.concatenate([zc , zc ])            

        n_cp += len(xa1)        
        n_sp += len(xa1_ts)  
        # ---------------------------------------------------------------------------------------
        # STEP 7: Store wing in vehicle vector
        # ---------------------------------------------------------------------------------------               
        VD.XA1    = np.append(VD.XA1,xa1)
        VD.YA1    = np.append(VD.YA1,ya1)
        VD.ZA1    = np.append(VD.ZA1,za1)
        VD.XA2    = np.append(VD.XA2,xa2)
        VD.YA2    = np.append(VD.YA2,ya2)
        VD.ZA2    = np.append(VD.ZA2,za2)        
        VD.XB1    = np.append(VD.XB1,xb1)
        VD.YB1    = np.append(VD.YB1,yb1)
        VD.ZB1    = np.append(VD.ZB1,zb1)
        VD.XB2    = np.append(VD.XB2,xb2)                
        VD.YB2    = np.append(VD.YB2,yb2)        
        VD.ZB2    = np.append(VD.ZB2,zb2)     
                  
        VD.XA1_ts = np.append(VD.XA1_ts,xa1_ts)
        VD.YA1_ts = np.append(VD.YA1_ts,ya1_ts)
        VD.ZA1_ts = np.append(VD.ZA1_ts,za1_ts)
        VD.XA2_ts = np.append(VD.XA2_ts,xa2_ts)
        VD.YA2_ts = np.append(VD.YA2_ts,ya2_ts)
        VD.ZA2_ts = np.append(VD.ZA2_ts,za2_ts)        
        VD.XB1_ts = np.append(VD.XB1_ts,xb1_ts)
        VD.YB1_ts = np.append(VD.YB1_ts,yb1_ts)
        VD.ZB1_ts = np.append(VD.ZB1_ts,zb1_ts)
        VD.XB2_ts = np.append(VD.XB2_ts,xb2_ts)                
        VD.YB2_ts = np.append(VD.YB2_ts,yb2_ts)        
        VD.ZB2_ts = np.append(VD.ZB2_ts,zb2_ts)    
        VD.XC     = np.append(VD.XC ,xc)
        VD.YC     = np.append(VD.YC ,yc)
        VD.ZC     = np.append(VD.ZC ,zc)     
        
    VD.n_sw       = n_sw
    VD.n_cw       = n_cw 
    VD.n_cp       = n_cp  
    VD.n_sp       = n_sp  
    
    # Compute Panel Normals
    VD.normals = compute_unit_normal(VD)
    
    return VD 

    
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

def generate_nacelle_points(VD,component,propulsor,propulsor_name,tessellation):
    """ This generates the coordinate points on the surface of the fuselage

    Assumptions:
    None

    Source:
    None

    Inputs:
    VD                   - vortex distribution

    Properties Used:
    N/A


    """
    
    if propulsor_name == 'propeller': 
        origin = propulsor.propeller.origin  
    elif propulsor_name == 'rotor': 
        origin = propulsor.rotor.origin
    else:           
        origin = propulsor.origin
        
    if isinstance(propulsor,Lift_Cruise):
        if propulsor_name == 'propeller':
            l      = propulsor.propeller_engine_length
            h      = propulsor.propeller_nacelle_diameter/2 
            end    = propulsor.propeller_nacelle_end
            start  = propulsor.propeller_nacelle_start
            offset = propulsor.propeller_nacelle_offset
            

        if propulsor_name == 'rotor':
            l      = propulsor.rotor_engine_length
            h      = propulsor.rotor_nacelle_diameter/2 
            end    = propulsor.rotor_nacelle_end
            start  = propulsor.rotor_nacelle_start
            offset = propulsor.rotor_nacelle_offset
    else:
        h      = propulsor.nacelle_diameter/2
        l      = propulsor.engine_length  
        end    = propulsor.nacelle_end
        start  = propulsor.nacelle_start
        offset = propulsor.nacelle_offset
        
    elipse_length = l/(end-start)

    x             = np.linspace(-elipse_length/2,elipse_length/2,10)
    y             = np.sqrt((1 - (x**2)/((elipse_length/2)**2))*(h**2))
    nac_height    = y[int(start*10) : int(end*10)]
    nac_loc       = x[int(start*10) : int(end*10)] + elipse_length*offset[0] 
    num_nac_segs  = len(nac_height)

    num_p   = len(origin)
    nac_pts = np.zeros((num_p,num_nac_segs,tessellation ,3))

  
    # Thrust Angle  
    if component.nacelle_angle != None:
        ta  = component.nacelle_angle  
    else: 
        if propulsor_name == 'propeller': 
            try:
                ta = propulsor.propeller_thrust_angle 
            except: 
                ta = propulsor.thrust_angle
        elif propulsor_name == 'rotor': 
            try:
                ta = propulsor.rotor_thrust_angle 
            except: 
                ta = propulsor.thrust_angle
        else:
            ta= propulsor.thrust_angle 
    
    for ip in range(num_p):
        for i_seg in range(num_nac_segs):
            theta    = np.linspace(0,2*np.pi,tessellation)
            a        = nac_height[i_seg]
            b        = nac_height[i_seg]
            r        = np.sqrt((b*np.sin(theta))**2  + (a*np.cos(theta))**2)
            nac_ypts = r*np.cos(theta)
            nac_zpts = r*np.sin(theta) + offset[2] 

            # Rotate to thrust angle
            X = np.cos(ta)*nac_loc[i_seg]  +  np.sin(ta)*nac_zpts
            Y = nac_ypts
            Z = -np.sin(ta)*nac_loc[i_seg]  +  np.cos(ta)*nac_zpts

            nac_pts[ip,i_seg,:,0] = X + origin[ip][0]
            nac_pts[ip,i_seg,:,1] = Y + origin[ip][1]
            nac_pts[ip,i_seg,:,2] = Z + origin[ip][2]

    # store points
    VD.NAC_SURF_PTS = nac_pts
    
    return VD

def plot_nacelle_geometry(axes,VD,face_color,edge_color,alpha):
    """ This plots a 3D surface of the nacelle of the propulsor

    Assumptions:
    None

    Source:
    None

    Inputs:
    VD                   - vortex distribution

    Properties Used:
    N/A
    """

    num_nac_segs = len(VD.NAC_SURF_PTS[0,:,0,0])
    tesselation  = len(VD.NAC_SURF_PTS[0,0,:,0])
    num_p        = len(VD.NAC_SURF_PTS[:,0,0,0])
    for ip in range(num_p):
        for i_seg in range(num_nac_segs-1):
            for i_tes in range(tesselation-1):
                X = [VD.NAC_SURF_PTS[ip,i_seg  ,i_tes  ,0],
                     VD.NAC_SURF_PTS[ip,i_seg  ,i_tes+1,0],
                     VD.NAC_SURF_PTS[ip,i_seg+1,i_tes+1,0],
                     VD.NAC_SURF_PTS[ip,i_seg+1,i_tes  ,0]]
                Y = [VD.NAC_SURF_PTS[ip,i_seg  ,i_tes  ,1],
                     VD.NAC_SURF_PTS[ip,i_seg  ,i_tes+1,1],
                     VD.NAC_SURF_PTS[ip,i_seg+1,i_tes+1,1],
                     VD.NAC_SURF_PTS[ip,i_seg+1,i_tes  ,1]]
                Z = [VD.NAC_SURF_PTS[ip,i_seg  ,i_tes  ,2],
                     VD.NAC_SURF_PTS[ip,i_seg  ,i_tes+1,2],
                     VD.NAC_SURF_PTS[ip,i_seg+1,i_tes+1,2],
                     VD.NAC_SURF_PTS[ip,i_seg+1,i_tes  ,2]]
                verts = [list(zip(X, Y, Z))]
                collection = Poly3DCollection(verts)
                collection.set_facecolor(face_color)
                collection.set_edgecolor(edge_color)
                collection.set_alpha(alpha)
                axes.add_collection3d(collection)

    return


def compute_unit_normal(VD):
    """ This computed the unit normal vector of each panel


    Assumptions: 
    None

    Source:
    None
    
    Inputs:   
    VD                   - vortex distribution    
    
    Properties Used:
    N/A
    """     

     # create vectors for panel
    P1P2 = np.array([VD.XB1 - VD.XA1,VD.YB1 - VD.YA1,VD.ZB1 - VD.ZA1]).T
    P1P3 = np.array([VD.XA2 - VD.XA1,VD.YA2 - VD.YA1,VD.ZA2 - VD.ZA1]).T

    cross = np.cross(P1P2,P1P3) 

    unit_normal = (cross.T / np.linalg.norm(cross,axis=1)).T

     # adjust Z values, no values should point down, flip vectors if so
    unit_normal[unit_normal[:,2]<0,:] = -unit_normal[unit_normal[:,2]<0,:]

    return unit_normal
