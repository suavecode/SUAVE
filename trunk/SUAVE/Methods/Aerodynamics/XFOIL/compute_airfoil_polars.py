## @ingroup Methods-Aerodynamics-XFOIL
#read_results.py
# 
# Created:  Mar 2019, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data , Units
import numpy as np

## @ingroup Methods-Aerodynamics-XFOIL

def  compute_airfoil_polars(propeller,conditions,airfoils):
     num_airfoils = len(airfoils)
     # unpack 
     Rh = propeller.hub_radius
     Rt = propeller.tip_radius
     n = len(propeller.chord_distribution)
     cm = propeller.chord_distribution[round(n*0.5)] 
     
     # read airfoil geometry  
     airfoil_data = read_airfoil_geometry(airfoils)
     
     AR = 2*(Rt - Rh)/cm
     
     # Get all of the coefficients for AERODAS wings
     AoA_sweep = np.linspace(-10,90,101)
     CL = np.zeros((num_airfoils,len(AoA_sweep)))
     CD = np.zeros((num_airfoils,len(AoA_sweep)))
      
     # AERODAS
     for i in range(num_airfoils):           
          # read airfoil polars 
          xfoil_cl , xfoil_cd , xfoil_aoa =  read_airfoil_polars(airfoils[i])
          
          # computing approximate zero lift aoa
          xfoil_cl_plus = xfoil_cl[xfoil_cl>0]
          idx_zero_lift = np.where(xfoil_cl == min(xfoil_cl_plus))[0][0]
          A0  = xfoil_aoa[idx_zero_lift]
          
          # computing approximate lift curve slope
          cl_range = xfoil_aoa[idx_zero_lift:idx_zero_lift+50]
          aoa_range = xfoil_cl[idx_zero_lift:idx_zero_lift+50]
          S1 = np.mean(np.diff(cl_range)/np.diff(aoa_range))  
          
          # max lift coefficent and associated aoa
          CL1max  = np.max(xfoil_cl) 
          idx_aoa_max_prestall_cl = np.where(xfoil_cl == CL1max)[0][0]
          ACL1  = xfoil_aoa[idx_aoa_max_prestall_cl]
          
          # max drag coefficent and associated aoa
          CD1max  = np.max(xfoil_cd) 
          idx_aoa_max_prestall_cd = np.where(xfoil_cd == CD1max)[0][0]
          ACD1   = xfoil_aoa[idx_aoa_max_prestall_cd]          
          
          CD0     = xfoil_cd[idx_zero_lift]       
          CL1maxp = CL1max
          ACL1p   = ACL1
          ACD1p   = ACD1
          CD1maxp = CD1max
          S1p     = S1
          
          for j in range(len(AoA_sweep)):
               alpha  = AoA_sweep[j] 
               t_c = airfoil_data.thickness_to_chord[i]
  
               # Equation 5a
               ACL1   = ACL1p + 18.2*CL1maxp*(AR**(-0.9)) 
          
               # From McCormick
               S1 = S1p*AR/(2+np.sqrt(4+AR**2)) 
          
               # Equation 5c
               ACD1   =  ACD1p + 18.2*CL1maxp*(AR**(-0.9)) 
          
               # Equation 5d
               CD1max = CD1maxp + 0.280*(CL1maxp*CL1maxp)*(AR**(-0.9))
          
               # Equation 5e
               CL1max = CL1maxp*(0.67+0.33*np.exp(-(4.0/AR)**2.))
               
               # ------------------------------------------------------
               # Equations for coefficients in pre-stall regime 
               # ------------------------------------------------------
               # Equation 6c
               RCL1   = S1*(ACL1-A0)-CL1max
          
               # Equation 6d
               N1     = 1 + CL1max/RCL1
               
               # Equation 6a or 6b depending on the alpha                  
               if alpha == A0:
                    CL[i,j] = 0.0        
               elif alpha > A0: 
                    CL[i,j] = S1*(alpha - A0)-RCL1*((alpha-A0)/(ACL1-A0))**N1        
               else:
                    CL[i,j] = S1*(alpha - A0)+RCL1 *((A0-alpha )/(ACL1 -A0))**N1 
                    
               # Equation 7a or 7b depending on alpha
               M    = 2.0  
               con  = np.logical_and((2*A0-ACD1)<=alpha,alpha<=ACD1)
               if con == True:
                    CD[i,j] = CD0  + (CD1max -CD0)*((alpha  -A0)/(ACD1 -A0))**M   
               else:
                    CD[i,j] = 0.  
               # ------------------------------------------------------
               # Equations for coefficients in post-stall regime 
               # ------------------------------------------------------               
               # Equation 9a and b
               F1        = 1.190*(1.0-(t_c**2))
               F2        = 0.65 + 0.35*np.exp(-(9.0/AR)**2.3)
          
               # Equation 10b and c
               G1        = 2.3*np.exp(-(0.65*t_c)**0.9)
               G2        = 0.52 + 0.48*np.exp(-(6.5/AR)**1.1)
          
               # Equation 8a and b
               CL2max    = F1*F2
               CD2max    = G1*G2
          
               # Equation 11d
               RCL2      = 1.632-CL2max
          
               # Equation 11e
               N2        = 1 + CL2max/RCL2
          
               # LIFT COEFFICIENT
               # Equation 11a,b,c
               if alpha > ACL1:
                    con2      = np.logical_and(ACL1<=alpha,alpha<=(92.0))
                    con3      = [alpha>=(92.0)]                   
                    if con2 == True:
                         CL[i,j] = -0.032*(alpha-92.0) - RCL2*((92.-alpha)/(51.0))**N2
                    elif con3 == True:
                         CL[i,j] = -0.032*(alpha-92.0) + RCL2*((alpha-92.)/(51.0))**N2

               # If alpha is negative flip things for lift
               elif alpha < 0.:  
                    alphan    = - alpha+2*A0
                    con2      = np.logical_and(ACL1<=alpha, alpha<=(92.0))
                    con3      = alpha>=(92.0)                    
                    if con2 == True:
                         CL[i,j] = 0.032*(alphan-92.0) + RCL2*((92.-alpha)/(51.0))**N2
                    elif con3 == True:
                         CL[i,j] = 0.032*(alphan-92.0) - RCL2*((alphan-92.)/(51.0))**N2
               
               # DRAG COEFFICIENT
               # Equation 12a 
               if  alpha > ACD1:
                    CD[i,j]  = CD1max + (CD2max - CD1max) * np.sin(((alpha-ACD1)/(90.-ACD1))*90.*Units.degrees)
          
               # If alpha is negative flip things for drag
               elif alpha < 0.:
                    alphan    = -alpha + 2*A0
                    if alphan>=ACD1:
                         CD[i,j]  = CD1max + (CD2max - CD1max) * np.sin(((alphan-ACD1)/(90.-ACD1))*Units.degrees)     

                                        
     airfoil_data.cl_polars = CL
     airfoil_data.cd_polars = CD
     airfoil_data.aoa_sweep = AoA_sweep
     
     return airfoil_data  

def  read_airfoil_geometry(airfoils):
    
     """ This functions reads the results from the results text file created 
     by XFOIL 
     """  
     
     num_airfoils = len(airfoils)
     # unpack      
     
     airfoil_data = Data()
     airfoil_data.x_coordinates = []
     airfoil_data.y_coordinates = []
     airfoil_data.thickness_to_chord = []
     
     for i in range(num_airfoils):  
          fname = airfoils[i] + '.dat'
          # Open file and read column names and data block
          f = open(fname)
          
          # Ignore header
          for header_line in range(1):
               f.readline()     
          
          data_block = f.readlines()
          f.close()
          
          data_len = len(data_block)
          x_data = np.zeros(data_len)
          y_data = np.zeros(data_len)      
          
          # Loop through each value: append to each column
          for line_count , line in enumerate(data_block):
               x_data[line_count] = float(data_block[line_count][2:10].strip())
               y_data[line_count]  = float(data_block[line_count][11:20].strip())
               
          upper_surface =  y_data[0:int((data_len-1)/2)]
          lower_surface =  y_data[int((data_len+1)/2):data_len]
          thickness = upper_surface - lower_surface[::-1]
          airfoil_data.thickness_to_chord.append(np.max(thickness))    
          airfoil_data.x_coordinates.append(x_data)  
          airfoil_data.y_coordinates.append(y_data)     
          
     return airfoil_data

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
          xfoil_aoa[line_count] = float(data_block[line_count][2:8].strip())
          xfoil_cl[line_count]  = float(data_block[line_count][10:17].strip())
          xfoil_cd[line_count]  = float(data_block[line_count][20:27].strip())  
               
     return xfoil_cl , xfoil_cd , xfoil_aoa


