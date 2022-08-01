## @ingroup Core
# Data.py
#
# Created:  May 2022, E. Botero
# Modified: 

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data

from jax import numpy as jnp
import numpy as np

from collections.abc import Iterable


# ----------------------------------------------------------------------
#   Converters
# ----------------------------------------------------------------------

## @ingroup Core    
def to_numpy(obj):
    """This function will convert an abstract object containing JAX arrays into pure numpy arrays
    
        Assumptions:
        The object can be an array, a list, a tuple, string, or a SUAVE Data structure
    
        Source:
        N/A
    
        Inputs:
        None
    
        Outputs:
        None
    
        Properties Used:
        None
    """        
    
    # Check if the object is a numpy array, return the jnp version
    if isinstance(obj, jnp.ndarray) and not isinstance(obj,np.ndarray):
        obj = np.array(obj)

    # Check if the object is a numpy array, return the jnp version
    elif isinstance(obj,Data):
        obj = obj.do_recursive(to_numpy)    

    elif isinstance(obj,str):
        pass

    # Check if the object is iterable, if so iterate on using recursion this will work for lists and tuples but not dicts
    elif isinstance(obj,Iterable):
        try:
            obj = obj.__class__(map(to_numpy,obj))    
        except:
            pass
    
    return obj


## @ingroup Core
def to_jnumpy(obj):
    """This function will convert an abstract object containing numpy arrays into JAX arrays
    
        Assumptions:
        The object can be an array, a list, a tuple, string, or a SUAVE Data structure
    
        Source:
        N/A
    
        Inputs:
        None
    
        Outputs:
        None
    
        Properties Used:
        None
    """       
    
    # Check if the object is a numpy array, return the jnp version
    if isinstance(obj, np.ndarray) and not isinstance(obj,jnp.ndarray):
        obj = jnp.array(obj)
        
    # Check if the object is a numpy array, return the jnp version
    elif isinstance(obj,Data):
        obj = obj.do_recursive(to_jnumpy)    

    elif isinstance(obj,str):
        pass

    # Check if the object is iterable, if so iterate on using recursion this will work for lists and tuples but not dicts
    elif isinstance(obj,Iterable) and not isinstance(obj,jnp.ndarray):
        try:
            obj = obj.__class__(map(to_jnumpy,obj))    
        except:
            pass
    
    return obj
