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

def APC_11_x_4_7_Propeller():          
    prop                            = SUAVE.Components.Energy.Converters.Rotor()
    prop.inputs                     = Data() 
    prop.inputs.pitch_command       = 0 
    prop.inputs.y_axis_rotation     = 0.
    prop.tag                        = 'APC_11x4_7_Propeller'
    prop.tip_radius                 = (11/2)*Units.inches
    prop.hub_radius                 = prop.tip_radius*0.15 
    prop.number_of_blades           = 2  
    prop.thrust_angle               = 0. 
    dimensionless_radius_chord      = np.array([[0.1552847467328054, 0.11186397023752742 ],[0.2053658303920633, 0.13710483640179336 ],[0.2554278355432604, 0.15993608699799672 ],
                                                [0.3030954879328435, 0.1803605838023466  ],[0.3531145664409043, 0.19777019937040918 ],[0.4054803014404272, 0.21156252981016876 ],
                                                [0.4554326051702757, 0.22053849089001235 ],[0.505361060765048 , 0.2265024325097777  ],[0.5552656682247448, 0.2294543546694648  ],
                                                [0.6027568444147668, 0.22758990746923585 ],[0.6549651817227891, 0.2215029094724792  ],[0.7024038920156443, 0.21301201946007814 ],
                                                [0.7545454545454544, 0.1984913669751025  ],[0.8042831250596203, 0.18035915291424204 ],[0.8539826385576648, 0.15740770771725646 ],
                                                [0.9012877992940951, 0.13204950872841742 ],[0.953167032338071 , 0.0843966421825813  ],[0.99              , 0.03493942573690739 ]])
    dimensionless_radius_twist      = np.array([[0.1452380952380954, 19.68937875751503   ],[0.1976190476190478, 21.793587174348698  ],[0.25, 22.494989979959918                ],
                                                [0.2952380952380953, 21.8436873747495    ],[0.3500000000000001, 20.741482965931866  ],[0.4023809523809523, 19.138276553106213  ],
                                                [0.44761904761904736, 17.434869739478955 ],[0.5, 15.6312625250501                   ],[0.5500000000000003, 14.07815631262525   ],
                                                [0.6000000000000001, 12.725450901803606  ],[0.6499999999999999, 11.57314629258517   ],[0.6976190476190478, 10.470941883767534  ],
                                                [0.7476190476190476, 9.569138276553108   ], [0.7976190476190479, 8.667334669338675   ],[0.8500000000000001, 7.665330661322642   ],
                                                [0.8999999999999999, 6.6633266533066156  ],[0.9500000000000002, 5.360721442885772   ],[0.99, 3.9579158316633247                  ]])
             
    r_R                             = dimensionless_radius_chord[:,0]
    b_R                             = dimensionless_radius_chord[:,1] 
    beta                            = dimensionless_radius_twist[:,1]
    
    # estimate thickness 
    r_R_data                        = np.array([0.155,0.275,0.367,0.449,0.5,0.55,
                                                0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99])
    t_b_data                        = np.array([0.122,0.105,0.077,0.061,0.055,0.049,0.045,0.041,0.038
                                                ,0.035,0.033,0.031,0.029,0.027,0.026]) 
    b_D_data                        = np.array([0.14485,0.14587,0.1481,
                                                0.1499,0.15061,0.15058,0.14981,0.14831,0.1468,0.14529,0.14268,
                                                0.13764,0.12896,0.11304,0.085])    
    func_max_thickness_distribution = interp1d(r_R_data, t_b_data*b_D_data*2*prop.tip_radius, kind='cubic')   
    prop.max_thickness_distribution = func_max_thickness_distribution(r_R)  
    prop.twist_distribution         = beta*Units.degrees
    prop.chord_distribution         = b_R*prop.tip_radius    
    prop.radius_distribution        = r_R*prop.tip_radius      
    prop.mid_chord_alignment        = np.zeros_like(prop.chord_distribution)   
    prop.thickness_to_chord         = prop.max_thickness_distribution/prop.chord_distribution     
    ospath    = os.path.abspath(__file__)
    separator = os.path.sep
    rel_path  = os.path.dirname(ospath) + separator   
    airfoil                          = SUAVE.Components.Airfoils.Airfoil()   
    airfoil.coordinate_file          = rel_path +'../Airfoils/Clark_y.txt'
    airfoil.polar_files              = [rel_path +'../Airfoils/Polars/Clark_y_polar_Re_50000.txt',
                                      rel_path +'../Airfoils/Polars/Clark_y_polar_Re_100000.txt',rel_path +'../Airfoils/Polars/Clark_y_polar_Re_200000.txt',
                                      rel_path +'../Airfoils/Polars/Clark_y_polar_Re_500000.txt',rel_path +'../Airfoils/Polars/Clark_y_polar_Re_1000000.txt']
    airfoil.geometry                 = import_airfoil_geometry(airfoil.coordinate_file,airfoil.number_of_points)
    airfoil.polars                   = compute_airfoil_properties(airfoil.geometry,airfoil.polar_files)
    prop.append_airfoil(airfoil) 
    prop.airfoil_locations           = list(np.zeros(len(prop.radius_distribution)).astype(int)) 
    return prop