## @ingroup Components-Wings-Airfoils
# load_airfoils.py
# 
# Created:  
# Modified: Feb 2016, T. MacDonald

import os, glob
from .Airfoil import Airfoil
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil import import_airfoil_dat

## @ingroup Components-Wings-Airfoils
def load_airfoils(directory,extension='.dat'):
    """ Loads airfoil cooridinate points from .dat file
    
    Assumptions:
    None

    Source:
    N/A

    Inputs:
    None

    Outputs:
    None

    Properties Used:
    N/A
    """     
    
    pattern = '*' + extension
    
    Airfoils = {}
        
    for f in glob.glob(os.path.join(directory,pattern)):
        
        name = os.path.splitext(os.path.basename(f))[0]
        
        if name.startswith('_'):
            continue
        
        data = import_airfoil_dat(f)
        
        upper = [ [v[0],0.0,v[1]] for v in data['upper'] ]
        lower = [ [v[0],0.0,v[1]] for v in data['lower'] ]
        
        airfoil = Airfoil()
        
        airfoil.tag = name
        
        upper = airfoil.Curve(tag='upper',points=upper)
        lower = airfoil.Curve(tag='lower',points=lower)
        
        airfoil.Curves.append(upper)
        airfoil.Curves.append(lower)
        
        Airfoils[name] = airfoil
    
    return Airfoils
        