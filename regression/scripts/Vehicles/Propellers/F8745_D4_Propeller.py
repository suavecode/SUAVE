# F8745_D4_Propeller.py
#
# Created:  Oct 2022, M. Clarke

# Imports
import SUAVE
from SUAVE.Core import Units, Data  
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_airfoil_properties  import compute_airfoil_properties
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry \
     import import_airfoil_geometry    
from scipy.interpolate import interp1d
import os
import numpy as np   

# design propeller                                       
def F8745_D4_Propeller():  
    prop                            = SUAVE.Components.Energy.Converters.Propeller()
    prop.inputs                     = Data()
    prop.inputs.pitch_command       = 0 
    prop.inputs.y_axis_rotation     = 0.
    prop.tag                        = 'F8745_D4_Propeller'  
    prop.tip_radius                 = 2.03/2
    prop.hub_radius                 = prop.tip_radius*0.20
    prop.number_of_blades           = 2  
    r_R_data                        = np.array([ 0.2,0.300,0.450,0.601,0.747,0.901,0.950,0.975,0.998   ])    
    t_c_data                        = np.array([ 0.3585,0.1976,0.1148,0.0834,0.0648,0.0591,0.0562,0.0542,0.0533    ])    
    b_R_data                        = np.array([0.116,0.143,0.163,0.169,0.166,0.148,0.135,0.113,0.075  ])    
    beta_data                       = np.array([  0.362,0.286,0.216,0.170,0.135,0.112,0.105,0.101,0.098  ])* 100 
  
    num_sec = 30          
    new_radius_distribution         = np.linspace(0.2,0.98 ,num_sec)
    func_twist_distribution         = interp1d(r_R_data, (beta_data)* Units.degrees , kind='cubic')
    func_chord_distribution         = interp1d(r_R_data, b_R_data * prop.tip_radius , kind='cubic')
    func_radius_distribution        = interp1d(r_R_data, r_R_data * prop.tip_radius , kind='cubic')
    func_max_thickness_distribution = interp1d(r_R_data, t_c_data * b_R_data , kind='cubic')  
    
    prop.twist_distribution         = func_twist_distribution(new_radius_distribution)     
    prop.chord_distribution         = func_chord_distribution(new_radius_distribution)         
    prop.radius_distribution        = func_radius_distribution(new_radius_distribution)        
    prop.max_thickness_distribution = func_max_thickness_distribution(new_radius_distribution) 
    prop.thickness_to_chord         = prop.max_thickness_distribution/prop.chord_distribution 
    

    ospath    = os.path.abspath(__file__)
    separator = os.path.sep
    rel_path  = os.path.dirname(ospath) + separator     
    prop.airfoil_geometry            = [rel_path +'../Airfoils/Clark_y.txt']
    prop.airfoil_polars             = [[rel_path +'../Airfoils/Polars/Clark_y_polar_Re_50000.txt' ,
                                       rel_path +'../Airfoils/Polars/Clark_y_polar_Re_100000.txt',
                                       rel_path +'../Airfoils/Polars/Clark_y_polar_Re_200000.txt',
                                       rel_path +'../Airfoils/Polars/Clark_y_polar_Re_500000.txt',
                                       rel_path +'../Airfoils/Polars/Clark_y_polar_Re_1000000.txt',
                                       rel_path +'../Airfoils/Polars/Clark_y_polar_Re_3500000.txt',
                                       rel_path +'../Airfoils/Polars/Clark_y_polar_Re_5000000.txt']] 
    prop.airfoil_flag                     = True 

    prop.number_of_airfoil_section_points = 200   
     
    
    airfoil_geometry_data                               = import_airfoil_geometry(prop.airfoil_geometry[0],npoints = prop.number_of_airfoil_section_points) 
    airfoil_polar_data                                  = compute_airfoil_properties(airfoil_geometry_data, airfoil_polar_files= prop.airfoil_polars[0],use_pre_stall_data=True,linear_lift=True ) 
         
   
    prop.RE_data                                        = airfoil_polar_data.reynolds_numbers
    prop.aoa_data                                       = airfoil_polar_data.angle_of_attacks
    prop.airfoil_cl_surrogates                          = airfoil_polar_data.lift_coefficients
    prop.airfoil_cd_surrogates                          = airfoil_polar_data.drag_coefficients
    
    prop.airfoil_bl_aoa_data                            = airfoil_polar_data.boundary_layer_angle_of_attacks           
    prop.airfoil_bl_RE_data                             = airfoil_polar_data.boundary_layer_reynolds_numbers   
    prop.airfoil_bl_lower_surface_theta_surrogates      = airfoil_polar_data.boundary_layer_theta_lower_surface           
    prop.airfoil_bl_lower_surface_delta_surrogates      = airfoil_polar_data.boundary_layer_delta_lower_surface           
    prop.airfoil_bl_lower_surface_delta_star_surrogates = airfoil_polar_data.boundary_layer_delta_star_lower_surface          
    prop.airfoil_bl_lower_surface_Ue_surrogates         = airfoil_polar_data.boundary_layer_cf_lower_surface          
    prop.airfoil_bl_lower_surface_cf_surrogates         = airfoil_polar_data.boundary_layer_Ue_Vinf_lower_surface              
    prop.airfoil_bl_lower_surface_dp_dx_surrogates      = airfoil_polar_data.boundary_layer_dcp_dx_lower_surface          
    prop.airfoil_bl_upper_surface_theta_surrogates      = airfoil_polar_data.boundary_layer_theta_upper_surface           
    prop.airfoil_bl_upper_surface_delta_surrogates      = airfoil_polar_data.boundary_layer_delta_upper_surface           
    prop.airfoil_bl_upper_surface_delta_star_surrogates = airfoil_polar_data.boundary_layer_delta_star_upper_surface            
    prop.airfoil_bl_upper_surface_Ue_surrogates         = airfoil_polar_data.boundary_layer_cf_upper_surface          
    prop.airfoil_bl_upper_surface_cf_surrogates         = airfoil_polar_data.boundary_layer_Ue_Vinf_upper_surface            
    prop.airfoil_bl_upper_surface_dp_dx_surrogates      = airfoil_polar_data.boundary_layer_dcp_dx_upper_surface        
     
    prop.airfoil_polar_stations      = list(np.zeros(num_sec).astype(int))  
    prop.mid_chord_alignment         = np.zeros_like(prop.chord_distribution)  
        
    return prop