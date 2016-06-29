# Arrays.py
#
# Created:  Aug 2015, T. Lukacyzk
# Modified: Feb 2016, T. MacDonald
#           Jun 2016, E.Botero

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import numpy as np

# ----------------------------------------------------------------------
#   Array
# ----------------------------------------------------------------------       

array_type  = np.ndarray
matrix_type = np.matrixlib.defmatrix.matrix


def atleast_2d_col(A):
    return atleast_2d(A,'col')

def atleast_2d_row(A):
    return atleast_2d(A,'row')

def atleast_2d(A,oned_as='row'):
    ''' ensures A is an array and at least of rank 2
    '''
    
    # not an array yet
    if not isinstance(A,(array_type,matrix_type)):
        if not isinstance(A,(list,tuple)):
            A = [A]
        A = np.array(A)
        
    # check rank
    if np.rank(A) < 2:
        # expand row or col
        if oned_as == 'row':
            A = A[None,:]
        elif oned_as == 'col':
            A = A[:,None]
        else:
            raise Exception , "oned_as must be 'row' or 'col' "
            
    return A