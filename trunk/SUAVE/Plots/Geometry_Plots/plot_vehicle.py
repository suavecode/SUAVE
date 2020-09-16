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
    edge_color = 'black'
    alpha_val  = 0.5  
    
    # initalize figure 
    fig = plt.figure(save_filename) 
    axes = Axes3D(fig)    
    axes.view_init(elev= 20, azim= 210)  
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
        alpha      = 0.2
        
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
    
 
                        for propulsor in vehicle.propulsors:    
                            if 'propeller' in propulsor.keys(): 
                                prop =  propulsor.propeller
                                
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
                             
                                n_points   = 20
                                dim       = len(b)
                                num_props = len(prop.origin) 
                                theta     = np.linspace(0,2*np.pi,num_B+1)[:-1]   
                                
                                # create empty arrays for storing geometry
                                G = Data()
                                G.XA1 = np.zeros((num_props,num_B,dim-1,2*n_points-1))
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
                                            # iba - imboard airfoil section
                                            iba_max_t   = airfoil_data.thickness_to_chord[a_secl[j]]
                                            iba_xp      = rot*(- MCA[j] +  airfoil_data.x_coordinates[a_secl[j]]*b[j])             # x coord of airfoil
                                            iba_yp      = r[j]*np.ones_like(iba_xp)                                             # radial location        
                                            iba_zp      = (airfoil_data.y_coordinates[a_secl[j]]*(t[j]/iba_max_t)) # former airfoil y coord
                                            
                                            # rotation about y axis to create twist and position blade upright
                                            iba_trans_1 = [[np.cos(rot*flip_1 - rot*beta[j]  ),0 , -np.sin(rot*flip_1 - rot*beta[j])], [0 ,  1 , 0] , [np.sin(rot*flip_1 - rot*beta[j]) , 0 , np.cos(rot*flip_1 - rot*beta[j])]] 
                                        
                                            # rotation about x axis to create azimuth locations 
                                            iba_trans_2 = [[1 , 0 , 0],[0 , np.cos(theta[i] + rot*a_o + flip_2 ), np.sin(theta[i] + rot*a_o + flip_2)],[0,np.sin(theta[i] + rot*a_o + flip_2), np.cos(theta[i] + rot*a_o + flip_2)]] 
                        
                                            iba_trans   =  np.matmul(iba_trans_2,iba_trans_1)          
                                    
                                            # oba - outboard airfoil section 
                                            oba_max_t   = airfoil_data.thickness_to_chord[a_secl[j+1]]
                                            oba_xp      = - MCA[j+1] + airfoil_data.x_coordinates[a_secl[j+1]]*b[j+1]             # x coord of airfoil
                                            oba_yp      = r[j+1]*np.ones_like(oba_xp)                                                   # radial location        
                                            oba_zp      = airfoil_data.y_coordinates[a_secl[j+1]]*(t[j+1]/oba_max_t) # former airfoil y coord
                                            
                                            # rotation about y axis to create twist and position blade upright
                                            oba_trans_1 = [[np.cos(rot*flip_1 - rot*beta[j]  ),0 , -np.sin(rot*flip_1 - rot*beta[j])], [0 ,  1 , 0] , [np.sin(rot*flip_1 - rot*beta[j]) , 0 , np.cos(rot*flip_1 - rot*beta[j])]] 
                                        
                                            # rotation about x axis to create azimuth locations 
                                            oba_trans_2 = [[1 , 0 , 0],[0 , np.cos(theta[i] + rot*a_o + flip_2), np.sin(theta[i] + rot*a_o + flip_2)],[0,np.sin(theta[i] + rot*a_o + flip_2), np.cos(theta[i] + rot*a_o + flip_2)]] 
                                        
                                            oba_trans   =  np.matmul(oba_trans_2,oba_trans_1)   
                                    
                                            iba_x = np.zeros(len(iba_xp))
                                            iba_y = np.zeros(len(iba_yp))
                                            iba_z = np.zeros(len(iba_zp))                   
                                            oba_x = np.zeros(len(oba_xp))
                                            oba_y = np.zeros(len(oba_yp))
                                            oba_z = np.zeros(len(oba_zp))     
                                             
                                            for k in range(len(iba_yp)):
                                                iba_vec_1 = [[iba_xp[k]],[iba_yp[k]], [iba_zp[k]]]
                                                iba_vec_2  = np.matmul(iba_trans,iba_vec_1)
                                                
                                                iba_x[k] = iba_vec_2[0]
                                                iba_y[k] = iba_vec_2[1]
                                                iba_z[k] = iba_vec_2[2] 
                                                
                                                oba_vec_1 = [[oba_xp[k]],[oba_yp[k]], [oba_zp[k]]]
                                                oba_vec_2  = np.matmul(oba_trans,oba_vec_1)
                                                oba_x[k] = oba_vec_2[0]
                                                oba_y[k] = oba_vec_2[1]
                                                oba_z[k] = oba_vec_2[2]       
                                        
                                            # store points
                                            G.XA1[n_p,i,j,:] = iba_x[:-1] + prop.origin[n_p][0]
                                            G.YA1[n_p,i,j,:] = iba_y[:-1] + prop.origin[n_p][1] 
                                            G.ZA1[n_p,i,j,:] = iba_z[:-1] + prop.origin[n_p][2]
                                            G.XA2[n_p,i,j,:] = iba_x[1:]  + prop.origin[n_p][0]
                                            G.YA2[n_p,i,j,:] = iba_y[1:]  + prop.origin[n_p][1] 
                                            G.ZA2[n_p,i,j,:] = iba_z[1:]  + prop.origin[n_p][2]
                                                      
                                            G.XB1[n_p,i,j,:] = oba_x[:-1] + prop.origin[n_p][0]
                                            G.YB1[n_p,i,j,:] = oba_y[:-1] + prop.origin[n_p][1]  
                                            G.ZB1[n_p,i,j,:] = oba_z[:-1] + prop.origin[n_p][2]
                                            G.XB2[n_p,i,j,:] = oba_x[1:]  + prop.origin[n_p][0]
                                            G.YB2[n_p,i,j,:] = oba_y[1:]  + prop.origin[n_p][1]
                                            G.ZB2[n_p,i,j,:] = oba_z[1:]  + prop.origin[n_p][2]            
            
            
            # ------------------------------------------------------------------------
            # Plot Propellers 
            # ------------------------------------------------------------------------
            prop_face_color = 'red'
            prop_edge_color = 'red'
            prop_alpha      = 1
            
            num_prop = len(G.XA1[:,0,0,0])
            num_B    = len(G.XA1[0,:,0,0])
            num_sec  = len(G.XA1[0,0,:,0])
            num_surf = len(G.XA1[0,0,0,:])
            for p_idx in range(num_prop):  
                    for B_idx in range(num_B):
                        for sec in range(num_sec): 
                            for loc in range(num_surf): 
                                X = [G.XA1[p_idx,B_idx,sec,loc],
                                     G.XB1[p_idx,B_idx,sec,loc],
                                     G.XB2[p_idx,B_idx,sec,loc],
                                     G.XA2[p_idx,B_idx,sec,loc]]
                                Y = [G.YA1[p_idx,B_idx,sec,loc],
                                     G.YB1[p_idx,B_idx,sec,loc],
                                     G.YB2[p_idx,B_idx,sec,loc],
                                     G.YA2[p_idx,B_idx,sec,loc]]
                                Z = [G.ZA1[p_idx,B_idx,sec,loc],
                                     G.ZB1[p_idx,B_idx,sec,loc],
                                     G.ZB2[p_idx,B_idx,sec,loc],
                                     G.ZA2[p_idx,B_idx,sec,loc]]                    
                                prop_verts = [list(zip(X, Y, Z))]
                                prop_collection = Poly3DCollection(prop_verts)
                                prop_collection.set_facecolor(prop_face_color)
                                prop_collection.set_edgecolor(prop_edge_color) 
                                prop_collection.set_alpha(prop_alpha)
                                axes.add_collection3d(prop_collection)   
                    
                
            
    plt.axis('off') 
    plt.grid(None)  
    
    return   
 
  
