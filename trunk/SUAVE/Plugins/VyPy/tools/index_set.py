
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import numpy as np
from VyPy.tools.arrays import array_type, atleast_2d

from numpy import (
    array, zeros, ones, kron, hstack, vstack, arange, prod, flipud
)

# ----------------------------------------------------------------------
#   The Function
# ----------------------------------------------------------------------

def index_set(set_type,order,dim=None,constraint=None):
    """ Builds the multi-indicies for orthogonal polynomials
    
        I = index_set(type,order)
        I = index_set(type,order,dimension)
        # I = index_set(type,order,dimension,constraint) -- not implemented
    
        Constructs an array of nonnegative integers where each column of size
        'dimension' contains a multi-index corresponding to a product type
        multivariate orthogonal polynomial. 
       
        Inputs
          type:       A string dictating the type of basis set. The valid options
                      for type include: 'tensor' for a tensor product basis,
                      'full' or 'complete' or 'total order' for a full polynomial
                      basis, and 'constrained' for a custom constrained basis. 
          
          order:      For the cases of 'constrained' and 'full' (and equivalent)
                      types, the positive integer 'order' dictates the highest
                      order of polynomial in the basis. For type 'tensor', this
                      may be a vector of size 'dimension'.
       
        Optional inputs
          dimension:  If type is 'tensor' and 'order' is a scalar, then the
                      dimension of the multi-indices must be specified.
       
          constraint: A user-defined anonymous function that takes the form
                      @(index) constraint(index). It must take a valid
                      multi-index as its argument and return a number to be
                      compared to the given order. For example, the constraint
                      for the full polynomial basis is @(index) sum(index).
       
        Example:
          % construct a total order basis
          I = index_set('total order',4,2);
          P = pmpack_problem('twobytwo',[0.2,0.6]);
          X = spectral_galerkin(P.A,P.b,P.s,I);
       
        See also SPECTRAL_GALERKIN   
       
        Copyright 2009-2010 David F. Gleich (dfgleic@sandia.gov) and Paul G. 
        Constantine (pconsta@sandia.gov)
       
        History
        -------
        :2010-06-14: Initial release, matlab
        :2014-08-07: Translated to python
    """
    
    order = atleast_2d(order)
    
    if dim is None:
        dim = len(order)
    else:
        if len(order)>1 and dim != len(order):
            raise Exception , 'dim is not consistent with order'
    
    if set_type == 'tensor':
        if len(order)==1: 
            order=order* ones([dim,1])
        
        I= array([[1]])
        for i in range(dim):
            I = hstack([ kron(I, ones([order[i][0]+1,1]))                        , 
                         kron(ones([I.shape[0],1]), arange(order[i][0]+1)[None,:].T) ])
            
        I=I[:,1:]
        I=I.T
        
    elif set_type in ('full','complete','total order'):
        
        if ~len(order)==1:
            raise Exception , 'Order must be a scalar for constrained index set.'
        
        I=zeros([dim,1])
        
        for i in range(order[0,0]):
            II = full_index_set(i+1,dim)
            I  = hstack([I , II.T])
        
    elif set_type == 'constrained':
        
        raise NotImplementedError
        
        #if constraint=None or not len(constraint):
            #raise Exception , 'No constraint provided.'
        
        #if ~len(order)==1:
            #raise Exception , 'Order must be a scalar for constrained index set.'
        
        #if order[0,0]==0:
            #I = zeros(dim,1)
            
        #else:
            #I = array([[]])
            #limit = order[0,0]*ones(1,dim)+1;
            
            #basis = cell(1,length(limit));
            
            #for i in range(1,prod(limit)+1):
                #[basis{:}] = ind2sub(limit,i)
                
                #b = cell2mat(basis).T-1
                
                #if constraint(b) <= order[0,0]:
                    #I = hstack([I,b])
                
    else:
        raise Exception , 'Unrecognized type: %s' % set_type
    
    return I
    
def full_index_set(n,d):
    if d==1:
        I=np.array([[n]])
    else:
        I=np.ones([0,d])
        for i in range(n+1):
            II=full_index_set(n-i,d-1);
            m=II.shape[0]
            T= hstack([i*ones([m,1]),II])
            I= vstack([I,T])
    return I



if __name__ == '__main__':
    
    print flipud( index_set('tensor',2,2) )