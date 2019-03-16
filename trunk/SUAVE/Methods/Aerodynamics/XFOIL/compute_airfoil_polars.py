## @ingroup Methods-Aerodynamics-XFOIL
#read_results.py
# 
# Created:  Mar 2019, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Methods.Aerodynamics.AERODAS import post_stall_coefficients , pre_stall_coefficients , finite_aspect_ratio
from SUAVE.Core import Data
import numpy as np

## @ingroup Methods-Aerodynamics-XFOIL

def  compute_airfoil_polars(propeller,conditions,airfoils):
     num_airfoils = len(airfoil)
     # unpack 
     Rh = propeller.hub_radius
     Rt = propeller.tip_radius
     n = len(propeller.chord_distribution)
     cm = propeller.chord_distribution[round(n*0.5)] 
     
     state = Data()
     state.conditions = Data() 
     state.conditions.aerodynamics = Data()
     state.conditions.freestream.altitude
     
     settings = Data()
     
     geoemtry = Data()
     geometry.section = Data()
     geoemtry.tag = 'propeller'
     geoemtry.aspect_ratio = 2*(Rt - Rh)/cm
     geoemtry.vertical  = False  
     
      
     # AERODAS
     for i in range(len(num_airfoils)): 
          
          # read airfoil polars 
          xfoil_cl , xfoil_cd , xfoil_aoa =  read_airfoil_polars(airfoils[i])
          
          # computing approximate zero lift aoa
          xfoil_cl_plus = [xfoil_cl>0]
          idx_zero_lift = np.where(xfoil_cl == min(xfoil_cl_plus))[1]
          settings.section_zero_lift_angle_of_attack = xfoil_aoa[idx_zero_lift]
          
          # computing approximate lift curve slope
          cl_range = xfoil_aoa[idx_zero_lift,idx_zero_lift+50]
          aoa_range = xfoil_cl[idx_zero_lift,idx_zero_lift+50]
          lift_curve_slope = np.mean(np.diff(cl_range)/np.diff(aoa_range))  
          settings.section_lift_curve_slope          = lift_curve_slope 
          
          # max lift coefficent and associated aoa
          maximum_lift_coefficient  = np.max(xfoil_cl) 
          idx_aoa_max_prestall_cl = np.where(xfoil_cl == maximum_lift_coefficient)[1]
          aoa_max_prestall_cl = xfoil_aoa[idx_aoa_max_prestall_cl]
          
          # max drag coefficent and associated aoa
          maximum_drag_coefficient  = np.max(xfoil_cd) 
          idx_aoa_max_prestall_cd = np.where(xfoil_cd == maximum_drag_coefficient)[1]
          aoa_max_prestall_cd = xfoil_aoa[idx_aoa_max_prestall_cd]          
          
          geometry.section.angle_attack_max_prestall_lift   = aoa_max_prestall_cl
          geometry.pre_stall_maximum_drag_coefficient_angle = aoa_max_prestall_cd
          geometry.pre_stall_maximum_lift_coefficient       = maximum_lift_coefficient
          geometry.section.zero_lift_drag_coefficient       = xfoil_cd[idx_zero_lift]
          geometry.pre_stall_lift_curve_slope               = lift_curve_slope 
          geometry.pre_stall_maximum_lift_drag_coefficient  = maximum_drag_coefficient
          
          geometry.section.maximum_coefficient_lift                 = maximum_lift_coefficient
          geometry.section.zero_lift_drag_coefficient               = xfoil_cd[idx_zero_lift]
          geometry.section.angle_attack_max_prestall_lift           = aoa_max_prestall_cl
          geometry.section.pre_stall_maximum_drag_coefficient       = maximum_drag_coefficient
          geometry.section.pre_stall_maximum_drag_coefficient_angle = aoa_max_prestall_cd
          
          # Get all of the coefficients for AERODAS wings
          AoA_sweep = np.linspace(-30,100,131)
          for j in range(len(AoA_sweep)):
               state.conditions.aerodynamics.angle_of_attack = AoA_sweep[j] 
               CL1max, CD1max, S1, ACD1 = finite_aspect_ratio(state,settings,geometry)
               if AoA_sweep[j] < aoa_max_prestall_cl:
                    CL[i,j] , CD[i,j]  = pre_stall_coefficients(state,settings,geometry)
               else:
                    CL[i,j] , CD[i,j]  = post_stall_coefficients(state,settings,geometry)     
         
     airfoil_polars = Data(
     cl  = CL,
     cd  = CD,
     AoA = AoA_sweep, 
     )
     
     return airfoil_polars 

def  read_airfoil_polars(airfoil_name):
    
     """ This functions reads the results from the results text file created 
     by XFOIL 
     """  
     fname = airfoil_name + '_polar'
     # Open file and read column names and data block
     f = open(fname)
     # Ignore header
     for header_line in range(12):
          f.readline()     
     
     data_block = f.readlines()
     f.close()
     
     data_len = len(data_block)
     xfoil_aoa= np.zeros(data_len)
     xfoil_cl = np.zeros(data_len)
     xfoil_cd = np.zeros(data_len)     
     
     # Loop through each value: append to each column
     for line_count , line in enumerate(data_block):
          print(float(data_block[line_count][2:8].strip()))
          xfoil_aoa[line_count] = float(data_block[line_count][2:8].strip())
          xfoil_cl[line_count]  = float(data_block[line_count][10:17].strip())
          xfoil_cd[line_count]  = float(data_block[line_count][20:27].strip())  
               
     return xfoil_cl , xfoil_cd , xfoil_aoa
