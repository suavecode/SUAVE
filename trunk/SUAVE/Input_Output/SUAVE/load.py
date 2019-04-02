## @ingroup Input_Output-SUAVE
#load.py
#
# Created:  Jan 2015, T. Lukaczyk
# Modified: Nov 2016, T. MacDonald



# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import json
from SUAVE.Core import Data, DataOrdered
import numpy as np
from collections import OrderedDict

# ----------------------------------------------------------------------
#  Method
# ----------------------------------------------------------------------

## @ingroup Input_Output-SUAVE
def load(filename):
    """Converts a JSON file into a SUAVE data structure.

    Assumptions:
    JSON file was a previously saved SUAVE data structure.

    Source:
    N/A

    Inputs:
    filename   <string> - file to be loaded

    Outputs:
    data       SUAVE data structure

    Properties Used:
    N/A
    """ 
    
    # Get JSON string
    f = open(filename)
    res_string = f.readline()
    f.close()    
    
    # Convert to dictionary
    res_dict = json.loads(res_string,object_pairs_hook=OrderedDict)    
    
    # Convert to SUAVE data structure
    data = read_SUAVE_json_dict(res_dict)
    
    return data

## @ingroup Input_Output-SUAVE
def read_SUAVE_json_dict(res_dict):
    """Builds a SUAVE data structure based on a dictionary from a JSON file. This is initial case.

    Assumptions:
    Dictionary was created based on a previously saved SUAVE data structure.

    Source:
    N/A

    Inputs:
    res_dict    Dictionary based on the SUAVE data structure

    Outputs:
    SUAVE_data  SUAVE data structure

    Properties Used:
    N/A
    """      
    keys = res_dict.keys() # keys from top level
    SUAVE_data = Data() # initialize SUAVE data structure
    
    # Assign all values
    for k in keys:
        k = str(k)
        v = res_dict[k]
        SUAVE_data[k] = build_data_r(v) # recursive function
    return SUAVE_data

## @ingroup Input_Output-SUAVE
def build_data_r(v):
    """Builds a SUAVE data structure based on a dictionary from a JSON file. This is recursive step.

    Assumptions:
    Dictionary was created based on a previously saved SUAVE data structure.

    Source:
    N/A

    Inputs:
    v     generic value

    Outputs:
    ret   value converted to needed format

    Properties Used:
    N/A
    """          
    tv = type(v) # Get value type
    
    # Transform to SUAVE data structure with appropriate types
    if tv == OrderedDict:
        keys = v.keys()
        # Recursively assign values
        ret = DataOrdered()
        for k in keys:
            k = str(k)
            ret[k] = build_data_r(v[k])
    elif tv == list:
        ret = np.array(v)
    elif (tv == str): 
        ret = str(v)
    elif (tv == bool):
        ret = v
    elif tv == type(None):
        ret = None
    elif (tv == float) or (tv == int):
        ret = v        
    else:
        raise TypeError('Data type not expected in SUAVE JSON structure')

    return ret