
import numpy as np
import scipy as sp

from VyPy.tools import vector_distance, atleast_2d

def lhc_uniform(XB,NI,XI=None,maxits=100):
    ''' Latin Hypercube Sampling with uniform density
        iterates to maximize minimum L2 distance
    '''
    
    print "Latin Hypercube Sampling ... "
    
    # dimension
    ND = XB.shape[0]
    
    # initial points to respect
    if XI is None:
        XI = np.empty([0,ND])
       
    # output points
    XO = []
    
    # initialize
    mindiff = 0;
    
    # maximize minimum distance
    for it in range(maxits):
        
        # samples
        S = np.zeros([NI,ND])
        
        # populate samples
        for i_d in range(ND):
            
            # uniform distribution [0,1], latin hypercube binning
            S[:,i_d] = ( np.random.random([1,NI]) + np.random.permutation(NI) ) / NI
            
        # scale to hypercube bounds
        XS = S*(XB[:,1]-XB[:,0]) + XB[:,0]        
        
        # add initial points
        XX = np.vstack([ XI , XS ])
        
        # calc distances
        vecdiff = vector_distance(XX)[0]
        
        # update
        if vecdiff > mindiff:
            mindiff = vecdiff
            XO = XX
        
    #: for iterate
    
    print '  Minimum Distance = %.4g' % mindiff
    
    return XO