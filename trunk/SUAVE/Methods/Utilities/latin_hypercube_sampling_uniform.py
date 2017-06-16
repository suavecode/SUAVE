# latin_hypercube_sampling.py
#
# Created:  T. Lukaczyk (outside of SUAVE code)
# Modified: Apr 2017, M. Vegh


#by M. Vegh 

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import numpy as np

## If needed for mapping to normal distribution:
#from scipy.stats.distributions import norm 


# ----------------------------------------------------------------------
#   Latin Hypercube Sampling
# ----------------------------------------------------------------------
def latin_hypercube_sampling_uniform(variable_bounds,num_samples, max_iterations=100):
    
#def latin_hypercube_sampling(num_dimensions,num_samples,criterion='random'):
    ND = variable_bounds.shape[0]
    NI = num_samples
    # initial points to respect
    XI = np.empty([0,ND])
       
    # output points
    XO = []
    
    # initialize
    mindiff = 0;
    
    # maximize minimum distance
    for it in range(max_iterations):
        
        # samples
        S = np.zeros([NI,ND])
        
        # populate samples
        for i_d in range(ND):
            
            # uniform distribution [0,1], latin hypercube binning
            S[:,i_d] = ( np.random.random([1,NI]) + np.random.permutation(NI) ) / NI
            
        # scale to hypercube bounds
        XS = S*(variable_bounds[:,1]-variable_bounds[:,0]) + variable_bounds[:,0]        
        
        # add initial points
        XX = np.vstack([ XI , XS ])
        
        if max_iterations > 1:  
            
            # calc distances
            vecdiff = vector_distance(XX)[0]
            
            # update
            if vecdiff > mindiff:
                mindiff = vecdiff
                XO = XX
                
        else:
            XO = XX
        
    #: for iterate
    
    if max_iterations > 1:
        print '  Minimum Distance = %.4g' % mindiff
    x_out = XO
    return x_out

def vector_distance(x):
    ''' calculates distance between points in matrix X 
        with each other, or optionally to given point P
        returns min, max and matrix/vector of distances
    '''
    
    # distance matrix among X

    nK,nD = x.shape
    
    d = np.zeros([nK,nK,nD])
    for iD in range(nD):
        d[:,:,iD] = np.array([x[:,iD]])-np.array([x[:,iD]]).T
    D = np.sqrt( np.sum( d**2 , 2 ) )
    
    diag_inf = np.diag( np.ones([nK])*np.inf )
    dmin = np.min(np.min( D + diag_inf ))
    dmax = np.max(np.max( D ))
        

        
    return (dmin,dmax,D)