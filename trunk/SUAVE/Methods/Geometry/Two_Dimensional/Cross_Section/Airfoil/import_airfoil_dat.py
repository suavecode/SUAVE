## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
# import_airfoil_dat.py
# 
# Created:  
# Modified: Sep 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ------------------------------------------------------------
#  import airfoil dat
# ------------------------------------------------------------

## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
def import_airfoil_dat(filename):
    """Import an airfoil data file and stores it in a numpy array.
    
    Assumptions:
    Airfoil file in Lednicer format

    Source:
    None

    Inputs:
    filename   <string>

    Outputs:
    data       numpy array with airfoil data

    Properties Used:
    N/A
    """     
    
    filein = open(filename,'r')
    data = {}   
    data['header'] = filein.readline().strip() + filein.readline().strip()

    filein.readline()
    
    sections = ['upper','lower']
    data['upper'] = []
    data['lower'] = []
    section = None
    
    while True:
        line = filein.readline()
        if not line: 
            break
        
        line = line.strip()
        
        if line and section:
            pass
        elif not line and not section:
            continue
        elif line and not section:
            section = sections.pop(0)
        elif not line and section:
            section = None
            continue
        
        point = list(map(float,line.split()))
        data[section].append(point)
        
    for k,v in data.items():
        data[k] = np.array(v)
        
    return data