# Input_Output.SUAVE.load.py
#
# Created By:   Trent Jan 2015

""" Load a native SUAVE file """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core.Input_Output import load_data
import json
from SUAVE.Core import Data
import numpy as np
from collections import OrderedDict

# ----------------------------------------------------------------------
#  Method
# ----------------------------------------------------------------------

def load(filename):
    """ load data from file """
    
    f = open(filename)
    res_string = f.readline()
    f.close()    
    
    res_dict = json.loads(res_string,object_pairs_hook=OrderedDict)    
    
    data = read_SUAVE_json_dict(res_dict)
    
    return data

def read_SUAVE_json_dict(res_dict):
    keys = res_dict.keys()
    SUAVE_data = Data()
    for k in keys:
        v = res_dict[k]
        SUAVE_data[k] = build_data_r(v)
    return SUAVE_data

def build_data_r(v):
    tv = type(v)
    if tv == OrderedDict:
        keys = v.keys()
        ret = Data()
        for k in keys:
            ret[k] = build_data_r(v[k])
    elif tv == list:
        ret = np.array(v)
    elif (tv == unicode) or (tv == bool):
        ret = str(v)
    elif v == None:
        ret = None
    elif (tv == float) or (tv == int):
        ret = v        
    else:
        raise ValueError('Data type not expected in SUAVE JSON structure')
    
    return ret