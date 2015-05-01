
import numpy as np
import scipy as sp

from VyPy.tools import vector_distance, atleast_2d

def sub_sample(XS,NI,II=None,maxits=100):
    ''' sub sample an existing dataset
        iterates to maximize minimum L2 distance
    '''
    
    print "Monte Carlo SubSampling ... "
    
    # dimension
    NX,ND = XS.shape
    
    # initial points to respect
    if II is None:
        II = np.empty([0])    
    else:
        II = np.array(II)
       
    # output points
    XO = []
    IO = []
    
    # initialize
    mindiff = 0;
    
    # maximize minimum distance
    for it in range(maxits):
        
        i_d = np.random.permutation(NX)
        
        for i in II:
            i_d = i_d[i_d!=i]
            
        i_d = i_d[1:NI+1]
        
        i_d = np.hstack([II,i_d])
        
        # samples
        XX = XS[i_d,:]
        
        # calc distances
        vecdiff = vector_distance(XX)[0]
        
        # update
        if vecdiff > mindiff:
            mindiff = vecdiff
            XO = XX
            IO = i_d
        
    #: for iterate
    
    print '  Minimum Distance = %.4g' % mindiff
    
    return XO, IO