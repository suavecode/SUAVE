
import numpy as np
from numpy import cos, sin
from orientation_product import orientation_product

def angles_to_dcms(rotations,sequence='ZYX'):
    """ transform = angles_to_dcms([r1s,r2s,r3s],seq)
        builds euler angle rotation matrix
    
        Inputs:
            rotations = [r1s r2s r3s], column array of rotations
            sequence = 'ZYX' (default)
                       'ZXZ'
                       etc...
        Outputs:
            transform = 3-dimensional array with direction cosine matricies
                        patterned along dimension zero
    """
    
    # transform map
    Ts = {'X':Tx,'Y':Ty,'Z':Tz}
    
    # a bunch of eyes
    transform = new_tensor(rotations[:,0])
    
    # build the tranform
    for dim,angs in zip(sequence,rotations.T)[::-1]:
        transform = orientation_product( transform, Ts[dim](angs) )
    
    # done!
    return transform
  
def Tx(a):
    
    # T = np.array([[1,   0,  0],
    #               [0, cos,sin],
    #               [0,-sin,cos]])
    
    cos = np.cos(a)
    sin = np.sin(a)
                  
    T = new_tensor(a)
    
    T[:,1,1] = cos
    T[:,1,2] = sin
    T[:,2,1] = -sin
    T[:,2,2] = cos
    
    return T
        

def Ty(a):
    
    # T = np.array([[cos,0,-sin],
    #               [0  ,1,   0],
    #               [sin,0, cos]])
    
    cos = np.cos(a)
    sin = np.sin(a)     
    
    T = new_tensor(a)
    
    T[:,0,0] = cos
    T[:,0,2] = -sin
    T[:,2,0] = sin
    T[:,2,2] = cos
    
    return T

def Tz(a):
    
    # T = np.array([[cos ,sin,0],
    #               [-sin,cos,0],
    #               [0   ,0  ,1]])
        
    cos = np.cos(a)
    sin = np.sin(a)     
    
    T = new_tensor(a)
    
    T[:,0,0] = cos
    T[:,0,1] = sin
    T[:,1,0] = -sin
    T[:,1,1] = cos
        
    return T

def new_tensor(a):
    
    assert np.rank(a) == 1
    n_a = len(a)
    
    T = np.eye(3)
    T = np.resize(T,[n_a,3,3])    
    
    return T


# ------------------------------------------------------------
#   Module Tests
# ------------------------------------------------------------
if __name__ == '__main__':
    
    import numpy as np
    from orientation_transpose import orientation_transpose
    
    n_t = 5
    
    psi   = np.linspace(0,2,n_t)
    theta = np.linspace(0,-2,n_t)
    phi   = np.zeros([n_t])
    
    rotations = np.array([psi,theta,phi]).T
    
    Fx = np.linspace(0,10,n_t)
    Fy = np.linspace(0,10,n_t)
    Fz = np.linspace(0,10,n_t)
    
    F = np.array([Fx,Fy,Fz]).T
    
    print rotations
    print F
    print '\n'
    
    T = angles_to_dcms(rotations,'ZYX')
    
    print T
    print '\n'
    
    F2 = orientation_product(T,F)
    
    print F2
    print '\n'
    
    Tt = orientation_transpose(T)
    F3 = orientation_product(Tt,F2)
    
    print F3
    print '\n'
    
    print F - F3
    
    
    
    

    