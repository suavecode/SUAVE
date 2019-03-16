## @ingroup Methods-Aerodynamics-XFOIL
#read_results.py
# 
# Created:  Mar 2019, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Methods.Aerodynamics import AERODAS as Methods
from SUAVE.Core import Data
import numpy as np

## @ingroup Methods-Aerodynamics-XFOIL

def  compute_airfoil_polars(airfoil):
     xfoil_polars = read_airfoil_polars(airfoil)
     
     # AERODAS
     

     compute.setup_data = Methods.AERODAS_setup.setup_data

     # Get all of the coefficients for AERODAS wings
     compute.wings_coefficients = Process()
     compute.wings_coefficients = Process_Geometry('wings')
     compute.wings_coefficients.section_properties  = Methods.section_properties.section_properties
     compute.wings_coefficients.finite_aspect_ratio = Methods.finite_aspect_ratio.finite_aspect_ratio
     compute.wings_coefficients.pre_stall           = Methods.pre_stall_coefficients.pre_stall_coefficients
     compute.wings_coefficients.post_stall          = Methods.post_stall_coefficients.post_stall_coefficients     
     
     #    
    
     return

def  read_airfoil_polars(airfoil):
    
     """ This functions reads the results from the results text file created 
     by XFOIL 
     """  
    
     num_airfoils = len(airfoil)
     xfoil_polars = Data()
     xfoil_polars.aoa= np.zeros((num_airfoils,220))
     xfoil_polars.cl = np.zeros((num_airfoils,220))
     xfoil_polars.cd = np.zeros((num_airfoils,220))
     
     xfoil_aoa    = xfoil_polars.aoa
     xfoil_cl     = xfoil_polars.cl
     xfoil_cd     = xfoil_polars.cd    
     for i in range(num_airfoils):

          
          fname = airfoil[i] + '_polar'
          
          # Open file and read column names and data block
          f = open(fname)
          # Ignore header
          for header_line in range(12):
               f.readline()
                       
          data_block = f.readlines()
          f.close()
      
          # Loop through each value: append to each column
          for line_count , line in enumerate(data_block):
               print(float(data_block[line_count][2:8].strip()))
               xfoil_aoa[i,line_count] = float(data_block[line_count][2:8].strip())
               xfoil_cl[i,line_count]  = float(data_block[line_count][10:17].strip())
               xfoil_cd[i,line_count]  = float(data_block[line_count][20:27].strip())  
               
     return xfoil_polars
