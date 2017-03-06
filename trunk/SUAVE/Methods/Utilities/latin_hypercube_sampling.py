# latin_hypercube_sampling.py
#
# Created:  Jul 2016, R. Fenrich (outside of SUAVE code)
# Modified: Mar 2017, T. MacDonald

import numpy as np

## If needed for mapping to normal distribution:
#from scipy.stats.distributions import norm 

def latin_hypercube_sampling(num_dimensions,num_samples,criterion=''):
    
    n = num_dimensions
    samples = num_samples
    
    segsize = 1./samples
    lhd = np.zeros((samples,n))
    
    if( criterion == "" ): # sample is randomly chosen from within segment
        for jj in range(n):
            for ii in range(samples):
                segStart = ii*segsize
                lhd[ii,jj] = segStart + np.random.rand()*segsize
    elif( criterion == "center" ): # sample is chosen as center of segment
        for jj in range(n):
            for ii in range(samples):
                segStart = ii*segsize
                lhd[ii,jj] = segStart + 0.5*segsize        
    else:
        raise NotImplementedError("Other sampling criterion not implemented")
        
    # Randomly switch values around to create Latin Hypercube
    for jj in range(n):
        np.random.shuffle(lhd[:,jj])
        
    ## Map samples to the standard normal distribution (if needed for future functionality)
    #lhd = norm(loc=0,scale=1).ppf(lhd)

    return lhd

if __name__ == '__main__': 
    
    # Functionality Test and Use Example
    #
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
    
    pass
