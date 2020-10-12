## @ingroup Plots-Geometry_Plots
# plot_vehicle.py
# 
# Created:  Mar 2020, M. Clarke
#           Apr 2020, M. Clarke
#           Jul 2020, M. Clarke

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
from SUAVE.Methods.Geometry.Three_Dimensional \
     import  orientation_product, orientation_transpose

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
    VD = vehicle.vortex_distribution
    
    face_color = 'grey'        
    edge_color = 'grey'
    alpha_val  = 0.5  
    
    # initalize figure 
    fig = plt.figure(save_filename) 
    fig.set_size_inches(8, 8) 
    axes = Axes3D(fig)    
    axes.view_init(elev= 30, azim= 210)   
    
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
        
    if  plot_control_points:
        axes.scatter(VD.XC,VD.YC,VD.ZC, c='r', marker = 'o' )
 

    if 'Wake' in VD: 
        face_color = 'white'                
        edge_color = 'blue' 
        alpha      = 0.5
        
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
                        
    if np.all(VD.FUS_SURF_PTS) != None:   
        face_color = 'grey'                
        edge_color = 'black' 
        alpha      = 1
        
        num_fus_segs = len(VD.FUS_SURF_PTS[:,0,0])
        tesselation  = len(VD.FUS_SURF_PTS[0,:,0]) 
        for i_seg in range(num_fus_segs-1):
            for i_tes in range(tesselation-1):
                X = [VD.FUS_SURF_PTS[i_seg  ,i_tes  ,0],
                     VD.FUS_SURF_PTS[i_seg  ,i_tes+1,0],
                     VD.FUS_SURF_PTS[i_seg+1,i_tes+1,0],
                     VD.FUS_SURF_PTS[i_seg+1,i_tes  ,0]]
                Y = [VD.FUS_SURF_PTS[i_seg  ,i_tes  ,1],
                     VD.FUS_SURF_PTS[i_seg  ,i_tes+1,1],
                     VD.FUS_SURF_PTS[i_seg+1,i_tes+1,1],
                     VD.FUS_SURF_PTS[i_seg+1,i_tes  ,1]]
                Z = [VD.FUS_SURF_PTS[i_seg  ,i_tes  ,2],
                     VD.FUS_SURF_PTS[i_seg  ,i_tes+1,2],
                     VD.FUS_SURF_PTS[i_seg+1,i_tes+1,2],
                     VD.FUS_SURF_PTS[i_seg+1,i_tes  ,2]]                 
                verts = [list(zip(X, Y, Z))]
                collection = Poly3DCollection(verts)
                collection.set_facecolor(face_color)
                collection.set_edgecolor(edge_color) 
                collection.set_alpha(alpha)
                axes.add_collection3d(collection)  
    
 
    for propulsor in vehicle.propulsors:    
        if ('propeller' in propulsor.keys()) or ('rotor' in propulsor.keys()): 
            try:
                prop = propulsor.propeller
            except:
                prop = propulsor.rotor
            
            # ------------------------------------------------------------------------
            # Generate Propeller Geoemtry  
            # ------------------------------------------------------------------------
            
            # unpack
            Rt     = prop.tip_radius          
            Rh     = prop.hub_radius          
            num_B  = prop.number_blades       
            a_sec  = prop.airfoil_geometry          
            a_secl = prop.airfoil_polar_stations
            beta   = prop.twist_distribution         
            b      = prop.chord_distribution         
            r      = prop.radius_distribution 
            MCA    = prop.mid_chord_aligment
            t      = prop.max_thickness_distribution
            ta     = -propulsor.thrust_angle
         
            n_points  = 10
            af_pts    = (2*n_points)-1
            dim       = len(b)
            num_props = len(prop.origin) 
            theta     = np.linspace(0,2*np.pi,num_B+1)[:-1]   
            
            # create empty arrays for storing geometry
            G = Data()
            G.XA1 = np.zeros((dim-1,af_pts))
            G.YA1 = np.zeros_like(G.XA1)
            G.ZA1 = np.zeros_like(G.XA1)
            G.XA2 = np.zeros_like(G.XA1)
            G.YA2 = np.zeros_like(G.XA1)
            G.ZA2 = np.zeros_like(G.XA1)
            G.XB1 = np.zeros_like(G.XA1)
            G.YB1 = np.zeros_like(G.XA1)
            G.ZB1 = np.zeros_like(G.XA1)
            G.XB2 = np.zeros_like(G.XA1)
            G.YB2 = np.zeros_like(G.XA1)
            G.ZB2 = np.zeros_like(G.XA1)  
        
            for n_p in range(num_props):  
                rot    = prop.rotation[n_p] 
                a_o    = 0
                flip_1 = (np.pi/2)  
                flip_2 = (np.pi/2) 
                
                for i in range(num_B):   
                    # get airfoil coordinate geometry     
                    airfoil_data = import_airfoil_geometry(a_sec,npoints=n_points)   
                    
                    # store points of airfoil in similar format as Vortex Points (i.e. in vertices)
                    for j in range(dim-1): # loop through each radial station 
                        # --------------------------------------------
                        # INNER SECTION
                        # --------------------------------------------                        
                        # INNER SECTION POINTS    
                        ixpts = airfoil_data.x_coordinates[a_secl[j]]
                        izpts = airfoil_data.y_coordinates[a_secl[j]]
                                                
                        iba_max_t   = airfoil_data.thickness_to_chord[a_secl[j]]
                        iba_xp      = rot*(- MCA[j] + ixpts*b[j])             # x coord of airfoil
                        iba_yp      = r[j]*np.ones_like(iba_xp)                                             # radial location        
                        iba_zp      = izpts*(t[j]/iba_max_t) # former airfoil y coord 
        
                        iba_matrix = np.zeros((len(iba_zp),3))    
                        iba_matrix[:,0] = iba_xp
                        iba_matrix[:,1] = iba_yp
                        iba_matrix[:,2] = iba_zp
                        
                        # ROTATION MATRICES FOR INNER SECTION     
                        # rotation about y axis to create twist and position blade upright
                        iba_trans_1 = [[np.cos(rot*flip_1 - rot*beta[j]  ),0 , -np.sin(rot*flip_1 - rot*beta[j])],
                                       [0 ,  1 , 0] ,
                                       [np.sin(rot*flip_1 - rot*beta[j]) , 0 , np.cos(rot*flip_1 - rot*beta[j])]] 
                        
         
                        # rotation about x axis to create azimuth locations 
                        iba_trans_2 = [[1 , 0 , 0],
                                       [0 , np.cos(theta[i] + rot*a_o + flip_2 ), np.sin(theta[i] + rot*a_o + flip_2)],
                                       [0,np.sin(theta[i] + rot*a_o + flip_2), np.cos(theta[i] + rot*a_o + flip_2)]] 
                    
                        # roation about y to orient propeller/rotor to thrust angle 
                        iba_trans_3 = [[np.cos(ta),0 , -np.sin(ta)],
                                       [0 ,  1 , 0] ,
                                       [np.sin(ta) , 0 , np.cos(ta)]] 
                        
                        iba_trans  =  np.matmul(iba_trans_3,np.matmul(iba_trans_2,iba_trans_1))
                        irot_mat    = np.repeat(iba_trans[ np.newaxis,:,: ],len(iba_yp),axis=0)
                        
                        # --------------------------------------------
                        # OUTER SECTION
                        # -------------------------------------------- 
                        # OUTER SECTION POINTS    
                        oxpts = airfoil_data.x_coordinates[a_secl[j+1]]
                        ozpts = airfoil_data.y_coordinates[a_secl[j+1]]
                        
                        oba_max_t   = airfoil_data.thickness_to_chord[a_secl[j+1]]
                        oba_xp      = - MCA[j+1] + oxpts*b[j+1]   # x coord of airfoil
                        oba_yp      = r[j+1]*np.ones_like(oba_xp) # radial location        
                        oba_zp      = ozpts*(t[j+1]/oba_max_t)    # former airfoil y coord 
               
                        oba_matrix = np.zeros((len(oba_zp),3))     
                        oba_matrix[:,0] = oba_xp
                        oba_matrix[:,1] = oba_yp
                        oba_matrix[:,2] = oba_zp                        
                        
                        # ROTATION MATRICES FOR OUTER SECTION                         
                        # rotation about y axis to create twist and position blade upright
                        oba_trans_1 = [[np.cos(rot*flip_1 - rot*beta[j+1]  ),0 , -np.sin(rot*flip_1 - rot*beta[j+1])],
                                       [0 ,  1 , 0] ,
                                       [np.sin(rot*flip_1 - rot*beta[j+1]) , 0 , np.cos(rot*flip_1 - rot*beta[j+1])]]  
                                         
                        # rotation about x axis to create azimuth locations 
                        oba_trans_2 = [[1 , 0 , 0],
                                       [0 , np.cos(theta[i] + rot*a_o + flip_2), np.sin(theta[i] + rot*a_o + flip_2)],
                                       [0,np.sin(theta[i] + rot*a_o + flip_2), np.cos(theta[i] + rot*a_o + flip_2)]]  
                        
                        # roation about y to orient propeller/rotor to thrust angle 
                        oba_trans_3 = [[np.cos(ta),0 , -np.sin(ta)],
                                       [0 ,  1 , 0] ,
                                       [np.sin(ta) , 0 , np.cos(ta)]]   
                        
                        oba_trans  =  np.matmul(oba_trans_3,np.matmul(oba_trans_2,oba_trans_1))
                        orot_mat    = np.repeat(oba_trans[ np.newaxis,:,: ],len(oba_yp) , axis=0)
                 
                        # ---------------------------------------------------------------------------------------------
                        # ROTATE POINTS
                        iba_mat  = orientation_product(irot_mat,iba_matrix) 
                        oba_mat  = orientation_product(orot_mat,oba_matrix) 
                        
                        # ---------------------------------------------------------------------------------------------
                        # store points
                        G.XA1[j,:] = iba_mat[:-1,0] + prop.origin[n_p][0]
                        G.YA1[j,:] = iba_mat[:-1,1] + prop.origin[n_p][1] 
                        G.ZA1[j,:] = iba_mat[:-1,2] + prop.origin[n_p][2]
                        G.XA2[j,:] = iba_mat[1:,0]  + prop.origin[n_p][0]
                        G.YA2[j,:] = iba_mat[1:,1]  + prop.origin[n_p][1] 
                        G.ZA2[j,:] = iba_mat[1:,2]  + prop.origin[n_p][2]
                                         
                        G.XB1[j,:] = oba_mat[:-1,0] + prop.origin[n_p][0]
                        G.YB1[j,:] = oba_mat[:-1,1] + prop.origin[n_p][1]  
                        G.ZB1[j,:] = oba_mat[:-1,2] + prop.origin[n_p][2]
                        G.XB2[j,:] = oba_mat[1:,0]  + prop.origin[n_p][0]
                        G.YB2[j,:] = oba_mat[1:,1]  + prop.origin[n_p][1]
                        G.ZB2[j,:] = oba_mat[1:,2]  + prop.origin[n_p][2]   
                
    
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
                    
                
            
    plt.axis('off') 
    plt.grid(None)  
    
    return   
 
  
