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
#           Aug 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data  
import numpy as np
from scipy import interpolate

## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
def import_airfoil_geometry(airfoil_geometry_files, npoints = 100,surface_interpolation = 'cubic'):
    """This imports an airfoil geometry from a text file  and store
    the coordinates of upper and lower surfaces as well as the mean
    camberline
    
    Assumptions:
    Works for Selig and Lednicer airfoil formats. Automatically detects which format based off first line of data. Assumes it is one of those two.
    Source:
    airfoiltools.com/airfoil/index - method for determining format and basic error checking
    Inputs:
    airfoil_geometry_files   <list of strings>
    surface_interpolation   - type of interpolation used in the SciPy function. Preferable options are linear, quardratic and cubic. 
    Full list of options can be found here : 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d
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
    half_npoints = npoints//2        

    geometry                    = Data() 
    geometry.airfoil_names      = airfoil_geometry_files         
    geometry.x_coordinates      = np.zeros((num_airfoils,npoints))
    geometry.y_coordinates      = np.zeros((num_airfoils,npoints))
    geometry.thickness_to_chord = np.zeros((num_airfoils,1))
    geometry.max_thickness      = np.zeros((num_airfoils,1))
    geometry.camber_coordinates = np.zeros((num_airfoils,half_npoints))
    geometry.x_upper_surface    = np.zeros((num_airfoils,half_npoints))
    geometry.x_lower_surface    = np.zeros((num_airfoils,half_npoints))
    geometry.y_upper_surface    = np.zeros((num_airfoils,half_npoints))
    geometry.y_lower_surface    = np.zeros((num_airfoils,half_npoints))

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

        x_up_surf = np.array(x_up_surf)
        x_lo_surf = np.array(x_lo_surf)
        y_up_surf = np.array(y_up_surf)
        y_lo_surf = np.array(y_lo_surf)  
 
        # create custom spacing for more points and leading and trailing edge
        t            = np.linspace(0,4,npoints-1)
        delta        = 0.25 
        A            = 5
        f            = 0.25
        smoothsq     = 5 + (2*A/np.pi) *np.arctan(np.sin(2*np.pi*t*f + np.pi/2)/delta) 
        dim_spacing  = np.append(0,np.cumsum(smoothsq)/sum(smoothsq))
        
        # compute thickness, camber and concatenate coodinates 
        x_data        = np.hstack((x_lo_surf[::-1], x_up_surf[1:])) 
        y_data        = np.hstack((y_lo_surf[::-1], y_up_surf[1:]))   
        tck,u         = interpolate.splprep([x_data,y_data],k=3,s=0) 
        out           = interpolate.splev(dim_spacing,tck) 
        x_data        = out[0]   
        y_data        = out[1]  
        
        # shift points to leading edge (x = 0, y = 0)
        x_delta  = min(x_data)
        x_data   = x_data - x_delta 
        
        arg_min  = np.argmin(x_data) 
        y_delta  = y_data[arg_min]
        y_data   = y_data - y_delta
        
        if (x_data[arg_min] == 0) and (y_data[arg_min]  == 0): 
            x_data[arg_min]  = 0  
            y_data[arg_min]  = 0 
        
        # make sure points start and end at x = 1.0
        x_data[0]  = 1.0
        x_data[-1] = 1.0
        
        # make sure a small gap at trailing edge
        if (y_data[0] == y_data[-1]): 
            y_data[0]          = y_data[0]  - 1E-4
            y_data[-1]         = y_data[-1] + 1E-4
            
        # thicknes and camber distributions require equal points     
        x_up_surf_old  = np.array(x_up_surf)   
        arrx_up_interp = interpolate.interp1d(np.arange(x_up_surf_old.size),x_up_surf_old, kind=surface_interpolation)
        x_up_surf_new  = arrx_up_interp(np.linspace(0,x_up_surf_old.size-1,half_npoints))    
     
        x_lo_surf_old  = np.array(x_lo_surf) 
        arrx_lo_interp = interpolate.interp1d(np.arange(x_lo_surf_old.size),x_lo_surf_old, kind=surface_interpolation )
        x_lo_surf_new  = arrx_lo_interp(np.linspace(0,x_lo_surf_old.size-1,half_npoints)) 
     
        # y coordinate s 
        y_up_surf_old  = np.array(y_up_surf)   
        arry_up_interp = interpolate.interp1d(np.arange(y_up_surf_old.size),y_up_surf_old, kind=surface_interpolation)
        y_up_surf_new  = arry_up_interp(np.linspace(0,y_up_surf_old.size-1,half_npoints))    
     
        y_lo_surf_old  = np.array(y_lo_surf) 
        arry_lo_interp = interpolate.interp1d(np.arange(y_lo_surf_old.size),y_lo_surf_old, kind=surface_interpolation)
        y_lo_surf_new  = arry_lo_interp(np.linspace(0,y_lo_surf_old.size-1,half_npoints)) 
    
        # compute thickness, camber and concatenate coodinates 
        thickness      = y_up_surf_new - y_lo_surf_new
        camber         = y_lo_surf_new + thickness/2  
        max_t          = np.max(thickness)
        max_c          = max(x_data) - min(x_data)
        t_c            = max_t/max_c 
        
        geometry.thickness_to_chord[i] = t_c
        geometry.max_thickness[i]      = max_t     
        geometry.x_coordinates[i]      = x_data   
        geometry.y_coordinates[i]      = y_data      
        geometry.x_upper_surface[i]    = x_up_surf_new 
        geometry.x_lower_surface[i]    = x_lo_surf_new 
        geometry.y_upper_surface[i]    = y_up_surf_new 
        geometry.y_lower_surface[i]    = y_lo_surf_new              
        geometry.camber_coordinates[i] = camber
         
    return geometry