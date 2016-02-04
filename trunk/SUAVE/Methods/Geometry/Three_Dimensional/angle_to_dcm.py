
import autograd.numpy as np 
from numpy import cos, sin

def angle_to_dcm(rotation,sequence=(2,1,0),units='radians'):
    """ transform = angle_to_dcm([r1,r2,r3],seq)
        builds euler angle rotation matrix
    
        Inputs:
            rotation = [r1 r2 r3]
            sequence = (2,1,0) (default)
                       (2,1,2)
                       etc...
                       a combination of three indeces
        Outputs:
            transform = direction cosine transformation matrix
    """
    
    # transform map
    Ts = { 0:T0, 1:T1, 2:T2 }
    
    transform = np.diag([1.,1.,1.])
    for dim in sequence[::-1]:
        ang = rotations[dim]
        if units == 'degrees':
            ang = ang * np.pi/180.
        transform = np.dot(transform,Ts[dim](ang))
                
    return transform
  
def T0(a):
    return np.array([[1,      0,     0],
                     [0, cos(a),sin(a)],
                     [0,-sin(a),cos(a)]])

def T1(a):
    return np.array([[cos(a),0,-sin(a)],
                     [0     ,1,      0],
                     [sin(a),0, cos(a)]])

def T2(a):
    return np.array([[cos(a) ,sin(a),0],
                     [-sin(a),cos(a),0],
                     [0      ,0     ,1]])