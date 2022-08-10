## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
# import_airfoil_polars.py
# 
# Created:  Mar 2019, M. Clarke
#           Mar 2020, M. Clarke
#           Sep 2020, M. Clarke 
#           May 2021, R. Erhard
#           Nov 2021, R. Erhard
#           Jul 2022, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data  
import numpy as np

## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
def import_airfoil_polars(airfoil_polar_files, airfoil_names):
    """This imports airfoil polars from a text file output from XFOIL or from a 
    text file containing the (alpha, CL, CD) data from other sources.
    
    Assumptions:
    Input airfoil polars file is obtained from XFOIL or from Airfoiltools.com

    Source:
    N/A

    Inputs:
    airfoil polar files   <list of strings>

    Outputs:
    data       numpy array with airfoil data

    Properties Used:
    N/A
    """      
    
    # number of airfoils 
    num_airfoils = len(airfoil_polar_files)  
    
    num_polars   = 0
    for i in range(num_airfoils): 
        n_p = len(airfoil_polar_files[i])
        if n_p < 3:
            raise AttributeError('Provide three or more airfoil polars to compute surrogate')
        
        num_polars = max(num_polars, n_p)
    
    # create empty data structures    
    airfoil_raw_polar_data = Data()    
    airfoil_raw_polar_data.angle_of_attacks  = Data()
    airfoil_raw_polar_data.reynolds_number   = Data()
    airfoil_raw_polar_data.mach_number       = Data()
    airfoil_raw_polar_data.lift_coefficients = Data()
    airfoil_raw_polar_data.drag_coefficients = Data() 
    
    for i in range(num_airfoils): 
        Re, Ma, CL, CD, AoA = [], [], [], [], []
        
        for j in range(len(airfoil_polar_files[i])):   
        
            # check for xfoil format
            f = open(airfoil_polar_files[i][j]) 
            data_block = f.readlines()
            f.close()            
            
            if "XFOIL" in data_block[1]:
                xfoilPolarFormat = True
                header_idx = 10
            elif "xflr5" in data_block[0]:
                xfoilPolarFormat = True
                header_idx = 9
            else:
                xfoilPolarFormat = False
    
            # Read data          
            if xfoilPolarFormat:
                # get data, extract Re, Ma
                headers = data_block[header_idx].split()
                polarData = np.genfromtxt(airfoil_polar_files[i][j], encoding='UTF-8-sig', dtype=None, names=headers, skip_header=header_idx+2)
                infoLine = list(filter(lambda x: 'Re = ' in x, data_block))[0]
                
                ReString = str(float(infoLine.split('Re =')[1].split('e 6')[0]))
                MaString = str(float(infoLine.split('Mach =')[1].split(' Re')[0]))
            else:
                # get data, extract Re, Ma
                polarData = np.genfromtxt(airfoil_polar_files[i][j], delimiter=" ", encoding='UTF-8-sig', dtype=None, names=True)
                headers = polarData.dtype.names
                
                ReString = airfoil_polar_files[i][j].split('Re_',1)[1].split('e6',1)[0]
                MaString = airfoil_polar_files[i][j].split('Ma_',1)[1].split('_',1)[0]
            
            airfoil_aoa = polarData[headers[np.where(np.array(headers) == 'alpha')[0][0]]]
            airfoil_cl = polarData[headers[np.where(np.array(headers) == 'CL')[0][0]]]
            airfoil_cd = polarData[headers[np.where(np.array(headers) == 'CD')[0][0]]]           
        
            Re.append( float (ReString) * 1e6 )
            Ma.append( float (MaString)       )       
            CL.append(  airfoil_cl      )
            CD.append(  airfoil_cd     )
            AoA.append( airfoil_aoa     )
            
        
        airfoil_raw_polar_data.angle_of_attacks[airfoil_names[i]]  = AoA
        airfoil_raw_polar_data.reynolds_number[airfoil_names[i]]   = Re
        airfoil_raw_polar_data.mach_number[airfoil_names[i]]       = Ma
        airfoil_raw_polar_data.lift_coefficients[airfoil_names[i]] = CL
        airfoil_raw_polar_data.drag_coefficients[airfoil_names[i]] = CD
    airfoil_raw_polar_data.airfoil_names = airfoil_names
    return airfoil_raw_polar_data 

