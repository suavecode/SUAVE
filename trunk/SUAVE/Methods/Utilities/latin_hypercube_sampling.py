# latin_hypercube_sampling.py
#
# Created:  Jul 2016, R. Fenrich (outside of SUAVE code)
# Modified: Apr 2017, T. MacDonald


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import numpy as np

## If needed for mapping to normal distribution:
#from scipy.stats.distributions import norm 


# ----------------------------------------------------------------------
#   Latin Hypercube Sampling
# ----------------------------------------------------------------------

def latin_hypercube_sampling(num_dimensions,num_samples,bounds=None,criterion='random'):
    
    n = num_dimensions
    samples = num_samples
    
    segsize = 1./samples
    lhd = np.zeros((samples,n))
    
    if( criterion == "random" ): # sample is randomly chosen from within segment
        segment_starts = np.arange(samples)*segsize
        lhd_base       = np.transpose(np.tile(segment_starts,(n,1)))
        lhd            = lhd_base + np.random.rand(samples,n)*segsize
    elif( criterion == "center" ): # sample is chosen as center of segment
        segment_starts = np.arange(samples)*segsize
        lhd_base       = np.transpose(np.tile(segment_starts,(n,1)))
        lhd            = lhd_base + 0.5*segsize           
    else:
        raise NotImplementedError("Other sampling criterion not implemented")
        
    # Randomly switch values around to create Latin Hypercube
    for jj in range(n):
        np.random.shuffle(lhd[:,jj])
        
    ## Map samples to the standard normal distribution (if needed for future functionality)
    #lhd = norm(loc=0,scale=1).ppf(lhd)
    
    if bounds != None:
        lower_bounds = bounds[0]
        upper_bounds = bounds[1]
        lhd = lhd*(upper_bounds-lower_bounds) + lower_bounds

    return lhd

# ----------------------------------------------------------------------
#   Functionality Test and Use Example 
# ----------------------------------------------------------------------

if __name__ == '__main__': 
     
    # 2D test is performed with samples chosen randomly in segment
    # 3D test is performed with samples chosen at the center of the segment
    
    num_2d_samples = 5
    num_3d_samples = 5
    
    # Imports for display
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    
    np.random.seed(0)
    
    
    # 2D Test Case
    
    fig = plt.figure("2D Test Case",figsize=(8,6))
    axes = plt.gca()
    
    lhd = latin_hypercube_sampling(2,num_2d_samples)
    
    x = lhd[:,0]
    y = lhd[:,1]
    
    axes.scatter(x,y)  
    axes.set_xticks(np.linspace(0,1,num_2d_samples+1))
    axes.set_yticks(np.linspace(0,1,num_2d_samples+1))
    axes.grid()
    
    
    # 3D Test Case
    
    fig = plt.figure("3D Test Case",figsize=(8,6))
    axes = plt.gca(projection='3d')    
    
    lhd = latin_hypercube_sampling(3,num_3d_samples,'center')
    
    x = lhd[:,0]
    y = lhd[:,1]
    z = lhd[:,2]
    
    axes.scatter(x,y,z)
    axes.set_xticks(np.linspace(0,1,num_3d_samples+1))
    axes.set_yticks(np.linspace(0,1,num_3d_samples+1))
    axes.set_zticks(np.linspace(0,1,num_3d_samples+1))
    
    
    # Display plots for both cases
    plt.show()    
    
    import time
    
    ti = time.time()
    latin_hypercube_sampling(40,10000)
    tf = time.time()
    print 'Time for 40D, 10000 samples: ' + str(tf-ti) + ' s'
    # 0.12 s on Surface Pro 3
    
    pass
