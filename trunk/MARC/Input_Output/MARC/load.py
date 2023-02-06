## @ingroup Input_Output-MARC
#load.py
#
# Created:  Jan 2015, T. Lukaczyk
# Modified: Nov 2016, T. MacDonald



# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import json
from MARC.Core import Data, DataOrdered
import numpy as np
from collections import OrderedDict

# ----------------------------------------------------------------------
#  Method
# ----------------------------------------------------------------------

## @ingroup Input_Output-MARC
def load(filename):
    """Converts a JSON file into a MARC data structure.

    Assumptions:
    JSON file was a previously saved MARC data structure.

    Source:
    N/A

    Inputs:
    filename   <string> - file to be loaded

    Outputs:
    data       MARC data structure

    Properties Used:
    N/A
    """ 
    
    # Get JSON string
    f = open(filename)
    res_string = f.readline()
    f.close()    
    
    # Convert to dictionary
    res_dict = json.loads(res_string,object_pairs_hook=OrderedDict)    
    
    # Convert to MARC data structure
    data = read_MARC_json_dict(res_dict)
    
    return data

## @ingroup Input_Output-MARC
def read_MARC_json_dict(res_dict):
    """Builds a MARC data structure based on a dictionary from a JSON file. This is initial case.

    Assumptions:
    Dictionary was created based on a previously saved MARC data structure.

    Source:
    N/A

    Inputs:
    res_dict    Dictionary based on the MARC data structure

    Outputs:
    MARC_data  MARC data structure

    Properties Used:
    N/A
    """      
    keys = res_dict.keys() # keys from top level
    MARC_data = Data() # initialize MARC data structure
    
    # Assign all values
    for k in keys:
        k = str(k)
        v = res_dict[k]
        MARC_data[k] = build_data_r(v) # recursive function
    return MARC_data

## @ingroup Input_Output-MARC
def build_data_r(v):
    """Builds a MARC data structure based on a dictionary from a JSON file. This is recursive step.

    Assumptions:
    Dictionary was created based on a previously saved MARC data structure.

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
    
    # Transform to MARC data structure with appropriate types
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
        raise TypeError('Data type not expected in MARC JSON structure')

    return ret