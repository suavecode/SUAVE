## @ingroup Plots
# Vehicle_Plots.py
# 
# Created:  May 2018, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units 
import matplotlib.cm as cm
import matplotlib.pyplot as plt  
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

## @ingroup Plots
# ------------------------------------------------------------------
# Vortex Lattice Method Panelization 
# ------------------------------------------------------------------
def plot_vehicle_vlm_panelization(data, save_figure = False, save_filename = "VLM_Panelization"):     
    face_color = [0.9,0.9,0.9] # grey        
    edge_color = [0, 0, 0]     # black
    alpha_val = 0.5  
    fig = plt.figure(save_filename)
    axes = Axes3D(fig)    
    n_cp = data.n_cp 
    for i in range(n_cp): 
        X = [data.XA1[i],data.XB1[i],data.XB2[i],data.XA2[i]]
        Y = [data.YA1[i],data.YB1[i],data.YB2[i],data.YA2[i]]
        Z = [data.ZA1[i],data.ZB1[i],data.ZB2[i],data.ZA2[i]] 
        verts = [list(zip(X, Y, Z))]
        collection = Poly3DCollection(verts)
        collection.set_facecolor(face_color)
        collection.set_edgecolor(edge_color)
        collection.set_alpha(alpha_val)
        axes.add_collection3d(collection)     
        max_range = np.array([data.X.max()-data.X.min(), data.Y.max()-data.Y.min(), data.Z.max()-data.Z.min()]).max() / 2.0    
        mid_x = (data.X .max()+data.X .min()) * 0.5
        mid_y = (data.Y .max()+data.Y .min()) * 0.5
        mid_z = (data.Z .max()+data.Z .min()) * 0.5
        axes.set_xlim(mid_x - max_range, mid_x + max_range)
        axes.set_ylim(mid_y - max_range, mid_y + max_range)
        axes.set_zlim(mid_z - max_range, mid_z + max_range)          
        
    axes.scatter(data.XC,data.YC,data.ZC, c='r', marker = 'o' )
    plt.axis('off')
    plt.grid(None)
    return

# ------------------------------------------------------------------
#   Propeller Geoemtry 
# ------------------------------------------------------------------
def plot_propeller_geometry(prop, line_color = 'bo-', save_figure = False, save_filename = "Propeller_Geometry"):   
    # unpack
    Rt     = prop.tip_radius          
    Rh     = prop.hub_radius          
    num_B  = prop.number_blades       
    a_sec  = prop.airfoil_sections           
    a_secl = prop.airfoil_section_location   
    beta   = prop.twist_distribution         
    b      = prop.chord_distribution         
    r      = prop.radius_distribution        
    t      = prop.max_thickness_distribution
    
    # prepare plot parameters
    dim = len(b)
    theta = np.linspace(0,2*np.pi,num_B+1)

    fig = plt.figure(save_filename)
    fig.set_size_inches(10, 8)     
    axes = plt.axes(projection='3d') 
    axes.set_zlim3d(-1, 1)        
    axes.set_ylim3d(-1, 1)        
    axes.set_xlim3d(-1, 1)     
    
    chord = np.outer(np.linspace(0,1,10),b)
    for i in range(num_B):  
        # plot propeller planfrom
        surf_x = np.cos(theta[i]) * (chord*np.cos(beta)) - np.sin(theta[i]) * (r) 
        surf_y = np.sin(theta[i]) * (chord*np.cos(beta)) + np.cos(theta[i]) * (r) 
        surf_z = chord*np.sin(beta)                                
        axes.plot_surface(surf_x ,surf_y ,surf_z, color = 'gray')
        
        if  a_sec != None and a_secl != None:
            # check dimension of section  
            dim_sec = len(a_secl)
            if dim_sec != dim:
                raise AssertionError("Number of sections not equal to number of stations") 
                      
            # get airfoil coordinate geometry     
            airfoil_data = read_propeller_airfoils(a_sec)       
            
            #plot airfoils 
            for j in range(dim):
                airfoil_max_t = airfoil_data.thickness_to_chord[a_secl[j]]
                airfoil_xp = b[j] - airfoil_data.x_coordinates[a_secl[j]]*b[j]
                airfoil_yp = r[j]*np.ones_like(airfoil_xp)            
                airfoil_zp = airfoil_data.y_coordinates[a_secl[j]]*b[j]  * (t[j]/(airfoil_max_t*b[j]))
                
                transformation_1 = [[np.cos(beta[j]),0 , -np.sin(beta[j])], [0 ,  1 , 0] , [np.sin(beta[j]) , 0 , np.cos(beta[j])]]
                transformation_2 = [[np.cos(theta[i]) ,-np.sin(theta[i]), 0],[np.sin(theta[i]) , np.cos(theta[i]), 0], [0 ,0 , 1]] 
                transformation  = np.matmul(transformation_2,transformation_1)
                
                airfoil_x = np.zeros(len(airfoil_yp))
                airfoil_y = np.zeros(len(airfoil_yp))
                airfoil_z = np.zeros(len(airfoil_yp))     
                
                for k in range(len(airfoil_yp)):
                    vec_1 = [[airfoil_xp[k]],[airfoil_yp[k]], [airfoil_zp[k]]]
                    vec_2  = np.matmul(transformation,vec_1)
                    airfoil_x[k] = vec_2[0]
                    airfoil_y[k] = vec_2[1]
                    airfoil_z[k] = vec_2[2]
                            
                axes.plot3D(airfoil_x, airfoil_y, airfoil_z, color = 'gray')
                
    if save_figure:
        plt.savefig(save_filename + ".png")  
                    
    return


