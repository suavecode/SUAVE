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
def import_airfoil_polars(airfoil_polar_files):
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
    
    all_Re, all_Ma, all_CL, all_CD, all_AoA = [], [], [], [], []
    
    for i in range(num_airfoils): 
        Re, Ma, CL, CD, AoA = [], [], [], [], []
        
        for j in range(len(airfoil_polar_files[i])):   
        
            # check for xfoil format
            f = open(airfoil_polar_files[i][j]) 
            data_block = f.readlines()
            f.close()            
            
            if "XFOIL" in data_block[1]:
                xfoilPolarFormat = True
            else:
                xfoilPolarFormat = False
    
            # Read data          
            if xfoilPolarFormat:
                # get data, extract Re, Ma
                headers = data_block[10].split()
                polarData = np.genfromtxt(airfoil_polar_files[i][j], encoding='UTF-8-sig', dtype=None, names=headers, skip_header=12)
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
            
        all_Re.append(Re)    
        all_Ma.append(Ma) 
        all_CL.append(CL)
        all_CD.append(CD)  
        all_AoA.append(AoA)
        
    airfoil_raw_polar_data.angle_of_attacks  = np.array(all_AoA)
    airfoil_raw_polar_data.reynolds_number   = np.array(all_Re)
    airfoil_raw_polar_data.mach_number       = np.array(all_Ma)
    airfoil_raw_polar_data.lift_coefficients = np.array(all_CL)
    airfoil_raw_polar_data.drag_coefficients = np.array(all_CD)     
     
    return airfoil_raw_polar_data 

