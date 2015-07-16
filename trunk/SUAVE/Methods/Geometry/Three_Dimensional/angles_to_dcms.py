
import numpy as np
from numpy import cos, sin
from orientation_product import orientation_product

def angles_to_dcms(rotations,sequence=(2,1,0)):
    """ transform = angles_to_dcms([r1s,r2s,r3s],seq)
        builds euler angle rotation matrix
    
        Inputs:
            rotations = [r1s r2s r3s], column array of rotations
            sequence = (2,1,0) (default)
                       (2,1,2)
                       etc...
                       a combination of three column indeces
        Outputs:
            transform = 3-dimensional array with direction cosine matricies
                        patterned along dimension zero
    """
    
    # transform map
    Ts = { 0:T0, 1:T1, 2:T2 }
    
    # a bunch of eyes
    transform = new_tensor(rotations[:,0])
    
    # build the tranform
    for dim in sequence[::-1]:
        angs = rotations[:,dim]
        transform = orientation_product( transform, Ts[dim](angs) )
    
    # done!
    return transform
  
def T0(a):
    
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
        

def T1(a):
    
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

def T2(a):
    
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
    
    if a.dtype is np.dtype('complex'):
        T = T + 0j
    
    T = np.resize(T,[n_a,3,3])
    
    return T


# ------------------------------------------------------------
#   Module Tests
# ------------------------------------------------------------
if __name__ == '__main__':
    
    import numpy as np
    from orientation_transpose import orientation_transpose
    
    n_t = 5
    
    phi   = np.zeros([n_t])
    theta = np.linspace(0,-2,n_t)
    psi   = np.linspace(0,2,n_t)
    
    rotations = np.array([phi,theta,psi]).T
    
    Fx = np.linspace(0,10,n_t)
    Fy = np.linspace(0,10,n_t)
    Fz = np.linspace(0,10,n_t)
    
    F = np.array([Fx,Fy,Fz]).T
    
    print rotations
    print F
    print '\n'
    
    T = angles_to_dcms(rotations,[2,1,0])
    
    print T
    print '\n'
    
    F2 = orientation_product(T,F)
    
    F2_expected = np.array(
        [[  0.        ,   0.        ,   0.        ],
         [  4.17578046,   0.99539256,   0.56749556],
         [  7.9402314 ,  -1.50584339,  -3.11209913],
         [  8.04794057,  -6.95068339,  -7.46114288],
         [  7.04074369, -13.25444263,  -8.64567399]]        
    )
    
    print F2
    print '\n'
    
    print 'should be nearly zero:'
    print np.sum(F2-F2_expected)
    print '\n'
    
    Tt = orientation_transpose(T)
    F3 = orientation_product(Tt,F2)
    
    print F3
    print '\n'
    
    print 'should be nearly zero:'
    print np.sum(F - F3)
    
    
    
    

    