
import numpy as np
from numpy import cos, sin

def angle2dcm(rotation,sequence='ZYX',units='radians'):
    """ dcm = angle2dcm([r1,r2,r3],seq)
        builds euler angle rotation matrix
    
        Inputs:
            rotation = [r1 r2 r3]
            sequence = 'ZYX' (default)
                       'ZXZ'
                       etc...
        Outputs:
            transform = direction cosine transformation matrix
    """
    
    Ts = {'X':Tx,'Y':Ty,'Z':Tz}
    
    transform = np.diag([1.,1.,1.])
    for dim,ang in zip(sequence,rotation)[::-1]:
        if units == 'degrees':
            ang = ang * np.pi/180.
        transform = np.dot(transform,Ts[dim](ang))
        
    return transform
  
def Tx(a):
    return np.array([[1,      0,     0],
                     [0, cos(a),sin(a)],
                     [0,-sin(a),cos(a)]])

def Ty(a):
    return np.array([[cos(a),0,-sin(a)],
                     [0     ,1,      0],
                     [sin(a),0, cos(a)]])

def Tz(a):
    return np.array([[cos(a) ,sin(a),0],
                     [-sin(a),cos(a),0],
                     [0      ,0     ,1]])