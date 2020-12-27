## @ingroup Plots-Geometry_Plots
# generate_propeller_geometry.py
# 
# Created:  Mar 2020, M. Clarke
#           Apr 2020, M. Clarke
#           Jul 2020, M. Clarke
#           Sep 2020, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
from SUAVE.Core import Data 
import numpy as np 
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry import import_airfoil_geometry
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_naca_4series import compute_naca_4series 

## @ingroup Plots-Geometry_Plots 
def generate_propeller_geometry(prop, angle_offset = 0):
    
    """This plots the geoemtry of a propeller or rotor

    Assumptions:
    None

    Source:
    None

    Inputs:
    SUAVE.Components.Energy.Converters.Propeller()

    Outputs: 
    Plots

    Properties Used:
    N/A	
    """	 
    
    # unpack 
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
 
    airfoil_pts   = 20
    dim           = len(b)
    num_props     = len(prop.origin) 
    theta         = np.linspace(0,2*np.pi,num_B+1)[:-1]  
    if any(r) == None:
        r = np.linspace(Rh,Rt, len(b))
    
    # create empty arrays for storing geometry
    G = Data()
    G.XA1 = np.zeros((num_props,num_B,dim-1,(2*airfoil_pts)-1))
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
        
        a_o    = prop.rotation[n_p]*angle_offset   
        flip_1 = (np.pi/2)*prop.rotation[n_p]          
        flip_2 = (np.pi/2)*prop.rotation[n_p]
        if prop.rotation[n_p] == -1:
            flip_3 = -np.pi
        else:
            flip_3 = 0
        
        for i in range(num_B):  
            # check if airfoils are defined 
            if a_sec != None and a_secl != None:
                # check dimension of section  
                dim_sec = len(a_secl)
                if dim_sec != dim:
                    raise AssertionError("Number of sections not equal to number of stations") 
        
                # get airfoil coordinate geometry     
                airfoil_data = import_airfoil_geometry(a_sec,npoints=airfoil_pts )       
            
            # if no airfoil defined, use NACA 0012
            else:
                airfoil_data = compute_naca_4series(0,0,12,npoints=(airfoil_pts*2)-2)
                a_secl       = np.zeros(dim).astype(int)
                
            # store points of airfoil in similar format as Vortex Points (i.e. in vertices)
            for j in range(dim-1): # loop through each radial station 
                # iba - imboard airfoil section
                iba_max_t   = airfoil_data.thickness_to_chord[a_secl[j]]
                iba_xp      = b[j] - MCA[j]- airfoil_data.x_coordinates[a_secl[j]]*b[j]             # x coord of airfoil
                iba_yp      = r[j]*np.ones_like(iba_xp)                                             # radial location        
                iba_zp      = airfoil_data.y_coordinates[a_secl[j]]*b[j]  * (t[j] /iba_max_t) # former airfoil y coord
                
                iba_trans_1 = [[np.cos(beta[j]+ flip_3 ),0 , -np.sin(beta[j]+ flip_3 )], [0 ,  1 , 0] , [np.sin(beta[j]+ flip_3 ) , 0 , np.cos(beta[j]+ flip_3 )]]
                
                iba_trans_2 = [[np.cos(theta[i] + a_o) ,-np.sin(theta[i] + a_o), 0],[np.sin(theta[i] + a_o) , np.cos(theta[i] + a_o), 0], [0 ,0 , 1]]   
                
                iba_trans_3 = [[1 , 0 , 0],[0 , np.cos(flip_1), np.sin(flip_1)],[0,np.sin(flip_1), np.cos(flip_1)]] 
                
                iba_trans_4 = [[np.cos(flip_2) , -np.sin(flip_2),0],[np.sin(flip_2),np.cos(flip_2),0],[0,0,1]]
                
                iba_trans   = np.matmul(iba_trans_4,np.matmul(iba_trans_3,np.matmul(iba_trans_2,iba_trans_1)))          
        
                # oba - outboard airfoil section 
                oba_max_t   = airfoil_data.thickness_to_chord[a_secl[j+1]]
                oba_xp      = b[j+1] - MCA[j+1]- airfoil_data.x_coordinates[a_secl[j+1]]*b[j+1]             # x coord of airfoil
                oba_yp      = r[j+1]*np.ones_like(oba_xp)                                             # radial location        
                oba_zp      = airfoil_data.y_coordinates[a_secl[j+1]]*b[j+1]  * (t[j+1]/oba_max_t) # former airfoil y coord
                
                oba_trans_1 = [[np.cos(beta[j+1] + flip_3 ),0 , -np.sin(beta[j+1] + flip_3 )], [0 ,  1 , 0] , [np.sin(beta[j+1] + flip_3 ) , 0 , np.cos(beta[j+1]+ flip_3 )]]
                
                oba_trans_2 = [[np.cos(theta[i] + a_o) ,-np.sin(theta[i] + a_o), 0],[np.sin(theta[i] + a_o) , np.cos(theta[i] + a_o), 0], [0 ,0 , 1]] 
                
                oba_trans_3 = [[1 , 0 , 0],[0 , np.cos(flip_1), np.sin(flip_1)],[0,np.sin(flip_1), np.cos(flip_1)]]  
                
                oba_trans_4 = [[np.cos(flip_2) , -np.sin(flip_2),0],[np.sin(flip_2),np.cos(flip_2),0],[0,0,1]]
                
                oba_trans   = np.matmul(oba_trans_4,np.matmul(oba_trans_3,np.matmul(oba_trans_2,oba_trans_1)))
        
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
        
    return G  