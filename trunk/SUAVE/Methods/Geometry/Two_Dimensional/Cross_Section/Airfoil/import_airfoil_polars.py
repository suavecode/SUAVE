## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
# import_airfoil_polars.py
# 
# Created:  Mar 2019, M. Clarke
#           Mar 2020, M. Clarke
#           Sep 2020, M. Clarke 
#           May 2021, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data  
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
    num_polars   = len(airfoil_polar_files[0]) 
    
    # create empty data structures 
    airfoil_data = Data()
    dim_aoa      = 89 # this is done to get an AoA discretization of 0.25
    CL           = np.zeros((num_airfoils,num_polars,dim_aoa))
    CD           = np.zeros((num_airfoils,num_polars,dim_aoa)) 
    Re           = np.zeros((num_airfoils,num_polars))
    
    AoA_interp = np.linspace(-6,16,dim_aoa) 
    
    for i in range(num_airfoils): 
    
        for j in range(num_polars):   
            # Open file and read column names and data block
            f = open(airfoil_polar_files[i][j]) 
            data_block = f.readlines()
            f.close()
            
            # Ignore header
            for header_line in range(len(data_block)):
                line = data_block[header_line]   
                if 'Re =' in line:    
                    Re[i,j] = float(line[25:40].strip().replace(" ", ""))
                if '---' in line:
                    data_block = data_block[header_line+1:]
                    break
                
            # Remove any extra lines at end of file:
            last_line = False
            while last_line == False:
                if data_block[-1]=='\n':
                    data_block = data_block[0:-1]
                else:
                    last_line = True
            
            data_len = len(data_block)
            airfoil_aoa= np.zeros(data_len)
            airfoil_cl = np.zeros(data_len)
            airfoil_cd = np.zeros(data_len)     
        
            # Loop through each value: append to each column
            for line_count , line in enumerate(data_block):
                airfoil_aoa[line_count] = float(data_block[line_count][0:8].strip())
                airfoil_cl[line_count]  = float(data_block[line_count][10:17].strip())
                airfoil_cd[line_count]  = float(data_block[line_count][20:27].strip())   
          
            CL[i,j,:] = np.interp(AoA_interp,airfoil_aoa,airfoil_cl)
            CD[i,j,:] = np.interp(AoA_interp,airfoil_aoa,airfoil_cd)       
                 
        airfoil_data.angle_of_attacks  = AoA_interp
        airfoil_data.reynolds_number   = Re
        airfoil_data.lift_coefficients = CL
        airfoil_data.drag_coefficients = CD      
     
    return airfoil_data 

