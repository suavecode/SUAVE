
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import os, time, errno

from DataBunch import DataBunch as dbunch

# -------------------------------------------------------------------
#  diffing a data
# -------------------------------------------------------------------  

def diff(A,B):
    
    keys = set([])
    keys.update( A.keys() )
    keys.update( B.keys() )
    
    result = type(A)()
    result.clear()
    
    for key in keys:
        va = A.get(key,None)
        vb = B.get(key,None)
        if isinstance(va,dbunch) and isinstance(vb,dbunch):
            sub_diff = diff(va,vb)
            if sub_diff:
                result[key] = sub_diff
        elif (isinstance(va,dbunch) or isinstance(vb,dbunch)) or not va==vb:
            result[key] = [va,vb]
        
    return result