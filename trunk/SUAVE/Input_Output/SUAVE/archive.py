# Input_Output.SUAVE.save.py
#
# Created By:   Trent Jan 2015

""" Save a native SUAVE file """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
from SUAVE.Core.Input_Output import save_data
import numpy as np
import types
import json
from collections import OrderedDict

# ----------------------------------------------------------------------
#  Method
# ----------------------------------------------------------------------

def archive(data,filename):
    """ archive data to file 
        down-converts all Data-type classes to Data, regardless of type
        helps future-proof the data to changing package structure
    """
    
    res_dict = build_dict_base(data)
    res_string = json.dumps(res_dict)
    f = open(filename,'w')   
    f.write(res_string)
    f.close()  
       

def build_dict_base(base):
    
    keys = base.keys()
    base_dict = OrderedDict()
    for k in keys:
        v = base[k]
        base_dict[k] = build_dict_r(v)
    return base_dict
    
def build_dict_r(v):
    tv = type(v)
    if (tv == np.ndarray) or (tv == np.float64):
        ret = v.tolist()
    elif (tv == str) or (tv == bool):
        ret = v
    elif tv == type(None):
        ret = None
    elif (tv == float) or (tv == int):
        ret = v
    elif tv == types.FunctionType:
        ret = None
    else:
        try:
            keys = v.keys()
        except:
            raise ValueError('Unexpected data type in SUAVE data structure')
        ret = OrderedDict()
        for k in keys:
            ret[k] = build_dict_r(v[k])        
    
    return ret