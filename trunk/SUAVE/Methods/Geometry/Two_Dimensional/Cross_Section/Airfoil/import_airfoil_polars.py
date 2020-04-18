## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
# import_airfoil_polars.py
# 
# Created:  Mar 2019, M. Clarke
#           Mar 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data , Units
import numpy as np

## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
def  import_airfoil_polars(airfoil_polar_files):
    """This imports airfoil polars from a text file output from XFOIL or Airfoiltools.com
    
    Assumptions:
    Input airfoil polars file is obtained from XFOIL or from Airfoiltools.com

    Source:
    http://airfoiltools.com/

    Inputs:
    airfoil polar files   <list of strings>

    Outputs:
    data       numpy array with airfoil data

    Properties Used:
    N/A
    """      
    
    # number of airfoils 
    num_airfoils = len(airfoil_polar_files) 
    
    # create empty data structures 
    airfoil_data = Data()
    AoA = []
    CL  = []
    CD  = []

    for i in range(num_airfoils):   
        # Open file and read column names and data block
        f = open(airfoil_polar_files[i]) 
        
        # Ignore header
        for header_line in range(12):
            f.readline()     
    
        data_block = f.readlines()
        f.close()
    
        data_len = len(data_block)
        airfoil_aoa= np.zeros(data_len)
        airfoil_cl = np.zeros(data_len)
        airfoil_cd = np.zeros(data_len)     
    
        # Loop through each value: append to each column
        for line_count , line in enumerate(data_block):
            airfoil_aoa[line_count] = float(data_block[line_count][2:8].strip())
            airfoil_cl[line_count]  = float(data_block[line_count][10:17].strip())
            airfoil_cd[line_count]  = float(data_block[line_count][20:27].strip())   
            
        AoA.append(airfoil_aoa)
        CL.append(airfoil_cl)
        CD.append(airfoil_cd)
       
    airfoil_data.angle_of_attacks   = AoA
    airfoil_data.lift_coefficients  = CL 
    airfoil_data.drag_coefficients  = CD 
       
    return airfoil_data


