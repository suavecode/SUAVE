# F8745_D4_Propeller.py
#
# Created:  Oct 2022, M. Clarke

# Imports
import MARC
from MARC.Core import Units, Data  
from MARC.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_airfoil_properties  import compute_airfoil_properties
from MARC.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry \
     import import_airfoil_geometry    
from scipy.interpolate import interp1d
import os
import numpy as np   

# design propeller                                       
def F8745_D4_Propeller():  
    prop                            = MARC.Components.Energy.Converters.Propeller()
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
    airfoil                          = MARC.Components.Airfoils.Airfoil()   
    airfoil.coordinate_file          = rel_path +'../Airfoils/Clark_y.txt'
    airfoil.polar_files              = [rel_path +'../Airfoils/Polars/Clark_y_polar_Re_50000.txt' ,
                                       rel_path +'../Airfoils/Polars/Clark_y_polar_Re_100000.txt',
                                       rel_path +'../Airfoils/Polars/Clark_y_polar_Re_200000.txt',
                                       rel_path +'../Airfoils/Polars/Clark_y_polar_Re_500000.txt',
                                       rel_path +'../Airfoils/Polars/Clark_y_polar_Re_1000000.txt']
    airfoil.geometry                 = import_airfoil_geometry(airfoil.coordinate_file,airfoil.number_of_points)
    airfoil.polars                   = compute_airfoil_properties(airfoil.geometry,airfoil.polar_files)
    prop.append_airfoil(airfoil) 
    prop.airfoil_polar_stations      = list(np.zeros(num_sec).astype(int))  
    prop.mid_chord_alignment         = np.zeros_like(prop.chord_distribution)  
        
    return prop