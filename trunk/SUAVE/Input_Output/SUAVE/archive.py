# archive.py
#
# Created:  Jan 2015, T. Lukaczyk
# Modified: Jun 2016, T. MacDonald


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
        converts a SUAVE data structure to a JSON file
    """
    
    # Create a dictionary structure with the results
    res_dict = build_dict_base(data)
    
    # Convert the dictionary to a JSON string
    res_string = json.dumps(res_dict)
    
    # Write results to a file
    f = open(filename,'w')   
    f.write(res_string)
    f.close()  
       

def build_dict_base(base):
    
    keys = base.keys() # keys from top level
    base_dict = OrderedDict() # initialize dictionary
    # Ordered is used because some post processing currently
    # relies on the segments being in order
    
    # Assign all values
    for k in keys:
        v = base[k]
        base_dict[k] = build_dict_r(v) # recursive function
    return base_dict
    
def build_dict_r(v):
    tv = type(v) # Get value type
    
    # Transform to basic python data type as appropriate
    if (tv == np.ndarray) or (tv == np.float64):
        ret = v.tolist()
    elif (tv == str) or (tv == bool):
        ret = v
    elif tv == type(None):
        ret = None
    elif (tv == float) or (tv == int):
        ret = v
    elif tv == types.FunctionType: # Functions cannot be stored
        ret = None
    else:
        # Assume other data types are SUAVE data types and check
        try:
            keys = v.keys()
        except:
            raise TypeError('Unexpected data type in SUAVE data structure')
        # Recursively assign values
        ret = OrderedDict()
        for k in keys:
            ret[k] = build_dict_r(v[k])        
    
    return ret