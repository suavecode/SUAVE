## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
# import_airfoil_polars.py
# 
# Created:  Mar 2019, M. Clarke
#           Mar 2020, M. Clarke
#           Sep 2020, M. Clarke 
#           May 2021, R. Erhard
#           Nov 2021, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data, Units 
import numpy as np

## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
def  import_airfoil_polars(airfoil_polar_files,angle_of_attack_discretization = 89):
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
    num_polars                   = 0 
    n_p = len(airfoil_polar_files)
    if n_p < 3:
        raise AttributeError('Provide three or more airfoil polars to compute surrogate') 
    num_polars            = max(num_polars, n_p)       
    
    # create empty data structures 
    airfoil_data = Data() 
    AoA          = np.zeros((num_polars,angle_of_attack_discretization))
    CL           = np.zeros((num_polars,angle_of_attack_discretization))
    CD           = np.zeros((num_polars,angle_of_attack_discretization)) 
    CM           = np.zeros((num_polars,angle_of_attack_discretization))
    Re           = np.zeros(num_polars)
    Ma           = np.zeros(num_polars)
    A0           = np.zeros(num_polars)
    
    AoA_interp = np.linspace(-6,16,angle_of_attack_discretization)  
    
    for j in range(len(airfoil_polar_files)):   
        # Open file and read column names and data block
        f = open(airfoil_polar_files[j]) 
        data_block = f.readlines()
        f.close()
        
        # Ignore header
        for header_line in range(len(data_block)):
            line = data_block[header_line]   
            if 'Re =' in line:    
                Re[j] = float(line[25:40].strip().replace(" ", ""))
            if 'Mach =' in line:    
                Ma[j] = float(line[7:20].strip().replace(" ", ""))    
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
        airfoil_cm = np.zeros(data_len)
    
        # Loop through each value: append to each column
        for line_count , line in enumerate(data_block):
            airfoil_aoa[line_count] = float(data_block[line_count][0:9].strip())
            airfoil_cl[line_count]  = float(data_block[line_count][9:18].strip())
            airfoil_cd[line_count]  = float(data_block[line_count][19:28].strip())   
            airfoil_cm[line_count] = float(data_block[line_count][38:47].strip())  
      
        AoA[j,:] = AoA_interp
        CL[j,:]  = np.interp(AoA_interp,airfoil_aoa,airfoil_cl)
        CD[j,:]  = np.interp(AoA_interp,airfoil_aoa,airfoil_cd)
        CM[j,:]  = np.interp(AoA_interp,airfoil_aoa,airfoil_cm)
        
        # Calculate zero lift angle of attack from raw data
        A0[j] = calculate_zero_lift_angle_of_attack(airfoil_cl, airfoil_aoa)
        
    airfoil_data.aoa_from_polar               = AoA*Units.degrees
    airfoil_data.re_from_polar                = Re
    airfoil_data.cl_from_polar                = airfoil_cl
    airfoil_data.mach_number                  = Ma
    airfoil_data.lift_coefficients            = CL
    airfoil_data.drag_coefficients            = CD  
    airfoil_data.pitching_moment_coefficients = CM
    airfoil_data.zero_lift_angle_of_attack    = A0
     
    return airfoil_data 


def calculate_zero_lift_angle_of_attack(airfoil_cl, airfoil_aoa):
    # computing approximate zero lift aoa
    airfoil_cl_plus      = airfoil_cl[airfoil_cl>0]
    idx_zero_lift        = np.where(airfoil_cl == min(airfoil_cl_plus))[0][0]
    airfoil_cl_crossing  = airfoil_cl[idx_zero_lift-1:idx_zero_lift+1]
    airfoil_aoa_crossing = airfoil_aoa[idx_zero_lift-1:idx_zero_lift+1]
    try:
        A0  = np.interp(0,airfoil_cl_crossing, airfoil_aoa_crossing)* Units.deg 
    except:
        A0 = airfoil_aoa[idx_zero_lift] * Units.deg 
    
    # limit for unrealistic zero-lift alpha; positive camber --> negative A0
    A0 = np.min([A0, 0.0])
        
    return A0