

import numpy as np

def complex_step(function,x0,step=1.e-10,dim=None):
    """ two cases - 
        x0 is a 1d list or array, return derivatives for all elements
        x0 is a 2d array, return derivatives for perturbations along columns
        step is a float, or list/array of complex step values (one for each dimension)
        returns array of derivatives, with one more dimension than the output of function
    """
    
    # operate on columns
    x0 = x0.T
    
    # number of variables
    nx = len(x0)
    
    # initialize steps
    if isinstance(step,(float,int)):
        step = [step] * nx
    step = np.array(step)
    assert len(step) == nx
        
    # helper function
    def evaluate_step(i):
        
        # complex step
        h = step[i]
        xh = np.array(x0,dtype=complex)
        xh[i] += np.complex(0,h)
        
        # evaluate
        deriv = np.imag( function(xh.T) ) / h
        
        return deriv
    #: def evaluate_step()
    
    # evaluate specified dimension
    if not dim is None:
        result = evaluate_step(dim)
        # yield result ??
        
    # evalute all dimensions
    else:
        # initialize derivatives
        derivatives = []        
    
        for i in range(nx):
            deriv = evaluate_step(i)
            derivatives.append(deriv)
        
        # return array of derivatives
        result = np.array(derivatives)
        
    return result
    
        
    
def forward_difference(function,x0,step=1.e-10,dim=None):
    """ two cases - 
        x0 is a 1d list or array, return derivatives for all elements
        x0 is a 2d array, return derivatives for perturbations along columns
        step is a float, or list/array of complex step values (one for each dimension)
        returns array of derivatives, with one more dimension than the output of function
    """
    
    # operate on columns
    x0 = x0.T
    
    # number of variables
    nx = len(x0)
    
    # initialize steps
    if isinstance(step,(float,int)):
        step = [step] * nx
    step = np.array(step)
    assert len(step) == nx
    
    # evaluate initial function
    central = function(x0.T)
        
    # helper function
    def evaluate_step(i):
        
        # complex step
        h = step[i]
        xh = x0.copy()
        xh[i] += h
        
        # evaluate
        deriv = ( function(xh.T) - central ) / h
        
        return deriv
    #: def evaluate_step()
    
    # evaluate specified dimension
    if not dim is None:
        result = evaluate_step(dim)
        # yield result ??
        
    # evalute all dimensions
    else:
        # initialize derivatives
        derivatives = []        
    
        for i in range(nx):
            deriv = evaluate_step(i)
            derivatives.append(deriv)
        
        # return array of derivatives
        result = np.array(derivatives)
        
    return result
        
    
        
def central_difference(function,x0,step=1.e-10,dim=None):
    """ two cases - 
        x0 is a 1d list or array, return derivatives for all elements
        x0 is a 2d array, return derivatives for perturbations along columns
        step is a float, or list/array of complex step values (one for each dimension)
        returns array of derivatives, with one more dimension than the output of function
    """
    
    # operate on columns
    x0 = x0.T
    
    # number of variables
    nx = len(x0)
    
    # initialize steps
    if isinstance(step,(float,int)):
        step = [step] * nx
    step = np.array(step)
    assert len(step) == nx
    
    ## evaluate initial function
    #central = function(x0.T)
        
    # helper function
    def evaluate_step(i):
        
        # complex step
        h = step[i]
        xhp = x0.copy()
        xhm = x0.copy()
        xhp[i] += h
        xhm[i] -= h
        
        # evaluate
        deriv = ( function(xhp.T) - function(xhm.T) ) / (2.*h)
        
        return deriv
    #: def evaluate_step()
    
    # evaluate specified dimension
    if not dim is None:
        result = evaluate_step(dim)
        # yield result ??
        
    # evalute all dimensions
    else:
        # initialize derivatives
        derivatives = []        
    
        for i in range(nx):
            deriv = evaluate_step(i)
            derivatives.append(deriv)
        
        # return array of derivatives
        result = np.array(derivatives)
        
    return result
            
        
        
        