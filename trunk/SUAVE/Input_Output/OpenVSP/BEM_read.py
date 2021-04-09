## @ingroup Methods-Propulsion
# bem_read.py
#
# Created:  Sep 2018, W. Maier
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Data, Units
import numpy as np

# ----------------------------------------------------------------------
#  Reading BEM files
# ----------------------------------------------------------------------
## @ingroup Methods-Propulsion
def BEM_read(bem_file, units_type='SI'):
    """Function that takes in a BEM file from OpenVSP, etc. 
    and returns propellor data. Note, this only works for VSP at the 
    moment and is not speed efficient. But could be extended for Qprop
    or Xrotor
    
    Assumptions:
    Radius and Chord are normalized by Radius
    
    Inputs:
    bem_file
    units_type  
        
    Outputs:
    
    results.
      num_sections [-]
      num_blades   [-]
      diameter     [units]
      center       [units]
      radius       [-]
      chord        [-] 
      twist        [deg] 
       
    Source:
    None
    
    """
    results = Data()
    results.radius = []
    results.chord  = []
    results.twist  = []    

    # Flag for type of BEM file
    # Set Flag here 
    
    # Setting up correct unit conversion
    if units_type == 'SI':
        units_factor = Units.meter * 1.
    else:
        units_factor = Units.foot * 1.
        
    # Opening files
    cont  = True
    table = False
    f = open(bem_file,'r')
    while cont:
        line = f.readline()
        
        if table == True:           
            temp = [float(s) for s in line.split(',') if s.count('.') == 1 ] 
            results.radius = np.append(results.radius, temp[0])
            results.chord = np.append(results.chord, temp[1])
            results.twist = np.append(results.twist, temp[2])
            if len(results.twist) == ns:
                cont = False
                break            
            
        if line.startswith('Num_Sections'):
            ns = [int(s) for s in line.split() if s.isdigit()][0]
            results.num_sections = ns
            
        if line.startswith('Num_Blade'):
            results.num_blade = [int(s) for s in line.split() if s.isdigit()][0] 
            
        if line.startswith('Diameter'):
            results.diameter = [float(s) for s in line.split() if s.count('.') == 1][0] * units_factor
            
        if line.startswith('Center'):
            line = line.replace(':',',')
            line = line.replace('\n','')
            results.origin = [float(s) * units_factor for s in line.split(',') if s.count('.') == 1 ] 
            
        if line.startswith('Normal'):
            line = line.replace(':',',')
            line = line.replace('\n','')
            results.normal = [float(s) for s in line.split(',') if s.count('.') == 1 ] 
        
        if line.startswith('Radius'):
            table = True
                    
    f.close()        
    
    results.twist = results.twist * Units.degrees
    
    if results.radius[ns-1] >= 1.0:
        results.radius = results.radius[0:ns-1]
        results.chord  = results.chord[0:ns-1]
        results.twist  = results.twist[0:ns-1]
    
         

    return results