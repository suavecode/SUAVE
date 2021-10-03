## @ingroup Core
# Arrays.py
#
# Created:  Aug 2015, T. Lukacyzk
# Modified: Feb 2016, T. MacDonald
#           Jun 2016, E. Botero
#           Jan 2020, M. Clarke
#           Oct 2021, E. Botero

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import numpy as np

# ----------------------------------------------------------------------
#   Array
# ----------------------------------------------------------------------       

array_type  = np.ndarray
matrix_type = np.matrixlib.defmatrix.matrix

## @ingroup Core
def atleast_2d_col(A):
    """Makes a 2D array in column format

    Assumptions:
    None

    Source:
    N/A

    Inputs:
    A      [1-D Array]

    Outputs:
    A      [2-D Array]

    Properties Used:
    N/A
    """       
    return atleast_2d(A,'col')

## @ingroup Core
def atleast_2d_row(A):
    """Makes a 2D array in row format

    Assumptions:
    None

    Source:
    N/A

    Inputs:
    A      [1-D Array]

    Outputs:
    A      [2-D Array]

    Properties Used:
    N/A
    """       
    return atleast_2d(A,'row')

## @ingroup Core
def atleast_2d(A,oned_as='row'):
    """ensures A is an array and at least of rank 2

    Assumptions:
    Defaults as row

    Source:
    N/A

    Inputs:
    A      [1-D Array]

    Outputs:
    A      [2-D Array]

    Properties Used:
    N/A
    """       
    
    # not an array yet
    if not isinstance(A,(array_type,matrix_type)):
        if not isinstance(A,(list,tuple)):
            A = [A]
        A = np.array(A)
        
    # check rank
    if A.ndim < 2:
        # expand row or col
        if oned_as == 'row':
            A = A[None,:]
        elif oned_as == 'col':
            A = A[:,None]
        else:
            raise Exception("oned_as must be 'row' or 'col' ")
            
    return A


## @ingroup Core
def append_array(A=None,B=None):
    """ A stacking operation used by merged to put together data structures

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        A [array]
        B [array]

        Outputs:
        array

        Properties Used:
        None
    """       
    if isinstance(A,array_type) and isinstance(B,array_type):
        return np.vstack([A,B])
    else:
        return None