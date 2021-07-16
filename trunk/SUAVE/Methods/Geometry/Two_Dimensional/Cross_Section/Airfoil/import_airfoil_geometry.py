## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
# import_airfoil_geometry.py
# 
# Created:  Mar 2019, M. Clarke
# Modified: Mar 2020, M. Clarke
#           Apr 2020, M. Clarke
#           Apr 2020, M. Clarke
#           May 2020, B. Dalman
#           Sep 2020, M. Clarke
#           May 2021, E. Botero
#           May 2021, R. Erhard
#           Jun 2021, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data  
import numpy as np
import scipy.interpolate as interp

## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
def  import_airfoil_geometry(airfoil_geometry_files, npoints = 100):
    """This imports an airfoil geometry from a text file  and stores
    the coordinates of upper and lower surfaces as well as the mean
    camberline
    
    Assumptions:
    Works for Selig and Lednicer airfoil formats. Automatically detects which format based off first line of data. Assumes it is one of those two.

    Source:
    airfoiltools.com/airfoil/index - method for determining format and basic error checking

    Inputs:
    airfoil_geometry_files   <list of strings>

    Outputs:
    airfoil_data.
        thickness_to_chord 
        x_coordinates 
        y_coordinates
        x_upper_surface
        x_lower_surface
        y_upper_surface
        y_lower_surface
        camber_coordinates  

    Properties Used:
    N/A
    """ 
    
    if isinstance(airfoil_geometry_files,str):
        print('import_airfoil_geometry was expecting a list of strings with absolute paths to airfoils')
        print('Attempting to change path string to list')
        airfoil_geometry_files = [airfoil_geometry_files]
 
    num_airfoils = len(airfoil_geometry_files)
    # unpack      

    airfoil_data                    = Data()
    airfoil_data.x_coordinates      = []
    airfoil_data.y_coordinates      = []
    airfoil_data.thickness_to_chord = []
    airfoil_data.max_thickness      = []
    airfoil_data.camber_coordinates = []
    airfoil_data.x_upper_surface    = []
    airfoil_data.x_lower_surface    = []
    airfoil_data.y_upper_surface    = []
    airfoil_data.y_lower_surface    = []
    
    for i in range(num_airfoils):  
        # Open file and read column names and data block
        f = open(airfoil_geometry_files[i]) 

        # Extract data
        data_block = f.readlines()
        try:
            # Check for header block
            first_element = float(data_block[0][0])
            if first_element == 1.:
                lednicer_format = False
        except:
            # Check for format line and remove header block
            format_line = data_block[1]
    
            # Check if it's a Selig or Lednicer file
            try:
                format_flag = float(format_line.strip().split()[0])
            except:
                format_flag = float(format_line.strip().split(',')[0])
                
            if format_flag > 1.01: # Amount of wiggle room per airfoil tools
                lednicer_format = True
                # Remove header block
                data_block      = data_block[3:]
            else:
                lednicer_format = False   
                # Remove header block
                data_block = data_block[1:]
                
        # Close the file
        f.close() 

        if lednicer_format:
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
                    x_up_surf.append(float(data_block[line_count].strip().split()[0])) 
                    y_up_surf.append(float(data_block[line_count].strip().split()[1])) 
                else:                              
                    x_lo_surf.append(float(data_block[line_count].strip().split()[0])) 
                    y_lo_surf.append(float(data_block[line_count].strip().split()[1]))   

        else:
            x_up_surf_rev  = []
            y_up_surf_rev  = []
            x_lo_surf      = []
            y_lo_surf      = []
            
            # Loop through each value: append to each column
            upper_surface_flag = True
            for line_count , line in enumerate(data_block): 
                #check for line which starts with 0., which should be the split between upper and lower in selig
                line_check = data_block[line_count].strip()
                
                # Remove any commas
                line_check = line_check.replace(',','')
                
                if float(line_check.split()[0]) == 0.:
                    x_up_surf_rev.append(float(data_block[line_count].strip().replace(',','').split()[0])) 
                    y_up_surf_rev.append(float(data_block[line_count].strip().replace(',','').split()[1]))

                    x_lo_surf.append(float(data_block[line_count].strip().replace(',','').split()[0])) 
                    y_lo_surf.append(float(data_block[line_count].strip().replace(',','').split()[1])) 

                    upper_surface_flag = False
                    continue

                if upper_surface_flag:
                    x_up_surf_rev.append(float(data_block[line_count].strip().replace(',','').split()[0])) 
                    y_up_surf_rev.append(float(data_block[line_count].strip().replace(',','').split()[1])) 
                else:                              
                    x_lo_surf.append(float(data_block[line_count].strip().replace(',','').split()[0])) 
                    y_lo_surf.append(float(data_block[line_count].strip().replace(',','').split()[1]))
                
                
                if upper_surface_flag ==True:
                    # check if next line flips without x-coordinate going to 0
                    next_line  = data_block[line_count+1].strip()
                    next_line  = next_line.replace(',','')
                
                    if next_line.split()[0]>line_check.split()[0] and next_line.split()[0] !=0.:
                        upper_surface_flag = False
                    
            # Upper surface values in Selig format are reversed from Lednicer format, so fix that
            x_up_surf_rev.reverse()
            y_up_surf_rev.reverse()

            x_up_surf = x_up_surf_rev
            y_up_surf = y_up_surf_rev


        # determine the thickness to chord ratio - note that the upper and lower surface
        # may be of different lenghts so initial interpolation is required 
        # x coordinates
        x_up_surf_old = np.array(x_up_surf)   
        arrx_up_interp= interp.interp1d(np.arange(x_up_surf_old.size),x_up_surf_old)
        x_up_surf_new = arrx_up_interp(np.linspace(0,x_up_surf_old.size-1,npoints))    
        
        x_lo_surf_old = np.array(x_lo_surf) 
        arrx_lo_interp= interp.interp1d(np.arange(x_lo_surf_old.size),x_lo_surf_old)
        x_lo_surf_new = arrx_lo_interp(np.linspace(0,x_lo_surf_old.size-1,npoints)) 
        
        # y coordinates 
        y_up_surf_old = np.array(y_up_surf)   
        arry_up_interp= interp.interp1d(np.arange(y_up_surf_old.size),y_up_surf_old)
        y_up_surf_new = arry_up_interp(np.linspace(0,y_up_surf_old.size-1,npoints))    
        
        y_lo_surf_old = np.array(y_lo_surf) 
        arry_lo_interp= interp.interp1d(np.arange(y_lo_surf_old.size),y_lo_surf_old)
        y_lo_surf_new = arry_lo_interp(np.linspace(0,y_lo_surf_old.size-1,npoints)) 
         
        # compute thickness, camber and concatenate coodinates 
        thickness     = y_up_surf_new - y_lo_surf_new
        camber        = y_lo_surf_new + thickness/2 
        x_data        = np.concatenate([x_up_surf_new[::-1],x_lo_surf_new])
        y_data        = np.concatenate([y_up_surf_new[::-1],y_lo_surf_new]) 
        
        max_t = np.max(thickness)
        max_c = max(x_data) - min(x_data)
        t_c   = max_t/max_c 
        
        airfoil_data.thickness_to_chord.append(t_c)
        airfoil_data.max_thickness.append(max_t)    
        airfoil_data.x_coordinates.append(x_data)  
        airfoil_data.y_coordinates.append(y_data)     
        airfoil_data.x_upper_surface.append(x_up_surf_new)
        airfoil_data.x_lower_surface.append(x_lo_surf_new)
        airfoil_data.y_upper_surface.append(y_up_surf_new)
        airfoil_data.y_lower_surface.append(y_lo_surf_new)          
        airfoil_data.camber_coordinates.append(camber)

    return airfoil_data 