## @ingroup Plots-Geometry_Plots
# plot_vehicle.py
# 
# Created:  Mar 2020, M. Clarke
#           Apr 2020, M. Clarke
#           Jul 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------  
import numpy as np 
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry \
     import import_airfoil_geometry 

## @ingroup Plots-Geometry_Plots
def plot_vehicle(vehicle, save_figure = False, plot_control_points = True, save_filename = "VLM_Panelization"):     
    """This plots vortex lattice panels created when Fidelity Zero  Aerodynamics 
    Routine is initialized

    Assumptions:
    None

    Source:
    None

    Inputs:
    airfoil_geometry_files   <list of strings>

    Outputs: 
    Plots

    Properties Used:
    N/A	
    """	
    
    VD = vehicle.vortex_distribution
    face_color = [0.9,0.9,0.9] # grey        
    edge_color = [0, 0, 0]     # black
    alpha_val  = 0.5  
    fig = plt.figure(save_filename)
    fig.set_size_inches(12, 12)
    axes = Axes3D(fig)    
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
            
            for p_idx in range(len(prop.origin)): 
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
                
                # prepare plot parameters
                dim = len(b)
                theta = np.linspace(0,2*np.pi,num_B+1)  
                chord = np.repeat(np.atleast_2d(b/2),10, axis = 0) - np.outer(np.linspace(0,1,10),b) - np.outer(np.linspace(0,1,10),MCA)
                  
                if r == None:
                    r = np.linspace(Rh,Rt, len(b))
                for i in range(num_B):  
                    
                    surf_x = np.cos(theta[i]) * (chord*np.cos(beta)) - np.sin(theta[i]) * (r) 
                    surf_y = np.sin(theta[i]) * (chord*np.cos(beta)) + np.cos(theta[i]) * (r) 
                    surf_z = chord*np.sin(beta)    
                    
                    # plot propeller planfrom
                    surf_X = prop.origin[p_idx][0] + surf_x  
                    surf_Y = prop.origin[p_idx][1] + np.cos(np.pi/2)*surf_y - np.sin(np.pi/2)*surf_z 
                    surf_Z = prop.origin[p_idx][2] + np.sin(np.pi/2)*surf_y + np.cos(np.pi/2)*surf_z 
                    
                    axes.plot_surface(surf_X ,surf_Y ,surf_Z , color = 'black')
                
                    if  a_sec != None and a_secl != None:
                        # check dimension of section  
                        dim_sec = len(a_secl)
                        if dim_sec != dim:
                            raise AssertionError("Number of sections not equal to number of stations") 
                
                        # get airfoil coordinate geometry     
                        airfoil_data = import_airfoil_geometry(a_sec)       
                
                        #plot airfoils 
                        for j in range(dim):
                            airfoil_max_t = airfoil_data.thickness_to_chord[a_secl[j]]
                            airfoil_xp = (b/2) - b[j] - MCA[j] - airfoil_data.x_coordinates[a_secl[j]]*b[j]
                            airfoil_yp = r[j]*np.ones_like(airfoil_xp)            
                            airfoil_zp = airfoil_data.y_coordinates[a_secl[j]]*b[j]  * (t[j]/(airfoil_max_t*b[j]))
                
                            transformation_1 = [[np.cos(beta[j]),0 , -np.sin(beta[j])], [0 ,  1 , 0] , [np.sin(beta[j]) , 0 , np.cos(beta[j])]]
                            transformation_2 = [[np.cos(theta[i]) ,-np.sin(theta[i]), 0],[np.sin(theta[i]) , np.cos(theta[i]), 0], [0 ,0 , 1]]    
                            transformation_3 = [[ 1 , 0 , 0] , [0 , np.cos(np.pi/2) ,-np.sin(np.pi/2)], [0 , np.sin(np.pi/2) , np.cos(np.pi/2)]]   
                            transformation  = np.matmul(transformation_3,np.matmul(transformation_2,transformation_1))
                
                            airfoil_x = np.zeros(len(airfoil_yp))
                            airfoil_y = np.zeros(len(airfoil_yp))
                            airfoil_z = np.zeros(len(airfoil_yp))     
                
                            for k in range(len(airfoil_yp)):
                                vec_1 = [[airfoil_xp[k]],[airfoil_yp[k]], [airfoil_zp[k]]]
                                vec_2  = np.matmul(transformation,vec_1)
                                airfoil_x[k] = prop.origin[p_idx][0] + vec_2[0]
                                airfoil_y[k] = prop.origin[p_idx][1] + vec_2[1]
                                airfoil_z[k] = prop.origin[p_idx][2] + vec_2[2]
                
                            axes.plot(airfoil_x, airfoil_y, airfoil_z, color = 'black')  
                
            
    plt.axis('off') 
    plt.grid(None)  
    
    return   
 
  
