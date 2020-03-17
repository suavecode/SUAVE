## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
# import_airfoil_geometry .py
# 
# Created:  Mar 2019, M. Clarke
#           Mar 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data , Units
import numpy as np
import scipy.interpolate as interp

## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
def  import_airfoil_geometry(airfoil_geometry_files):
    """This imports an airfoil geometry from a text file  and stores
    the coordinates of upper and lower surfaces as well as the mean ]
    camberline
    
    Assumptions:
    Airfoil file in Lednicer format

    Source:
    None

    Inputs:
    airfoil_geometry_files   <list of strings>

    Outputs:
    airfoil_data.
        thickness_to_chord 
        x_coordinates 
        y_coordinates  

    Properties Used:
    N/A
    """      
 
    num_airfoils = len(airfoil_geometry_files)
    # unpack      

    airfoil_data = Data()
    airfoil_data.x_coordinates = []
    airfoil_data.y_coordinates = []
    airfoil_data.thickness_to_chord = []

    for i in range(num_airfoils):  
        # Open file and read column names and data block
        f = open(airfoil_geometry_files[i]) 
                
        # Ignore header
        for header_line in range(3):
            f.readline()     

        data_block = f.readlines()
        f.close() 
        
        x_up_surf = []
        y_up_surf = []
        x_lo_surf = []
        y_lo_surf = []
        
        # Loop through each value: append to each column
        upper_surface_flag = True
        for line_count , line in enumerate(data_block): 
            #check for blank line which signifies the upper/lower surface division 
            line_check = data_block[line_count].strip()
            if line_check == '':
                upper_surface_flag = False
                continue
            if upper_surface_flag:
                x_up_surf.append(float(data_block[line_count][1:10].strip())) 
                y_up_surf.append(float(data_block[line_count][11:20].strip())) 
            else:                              
                x_lo_surf.append(float(data_block[line_count][1:10].strip())) 
                y_lo_surf.append(float(data_block[line_count][11:20].strip()))     
            
        x_data    = np.concatenate([np.array(x_up_surf) ,np.array(x_lo_surf)])
        y_data    = np.concatenate([np.array(y_up_surf) ,np.array(y_lo_surf)]) 
        
        
        # determine the thickness to chord ratio - not that the upper and lower surface
        # may be of differnt lenghts
        arr_ref      = np.array(y_up_surf)  
        arr2         = np.array(y_lo_surf)       
        arr2_interp  = interp.interp1d(np.arange(arr2.size),arr2)
        arr2_stretch = arr2_interp(np.linspace(0,arr2.size-1,arr_ref.size)) 
        thickness    = arr_ref - arr2_stretch
        
        airfoil_data.thickness_to_chord.append(np.max(thickness))    
        airfoil_data.x_coordinates.append(x_data)  
        airfoil_data.y_coordinates.append(y_data)     

    return airfoil_data 