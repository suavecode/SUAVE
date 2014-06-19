# test_NURBS_curve.py: test various NURBS curve capability

import SUAVE
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# MAIN
def main():
   
    # create a circle
    circle = SUAVE.Attributes.Geometry3D.Curve() 
    a = np.sqrt(2)/2
    circle.CPs.x = np.array([1, 1, 0, -1, -1, -1, 0, 1, 1])  
    circle.CPs.y = np.array([0, 1, 1, 1, 0, -1, -1, -1, 0])  
    circle.CPs.z = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])  
    circle.w = np.array([1, a, 1, a, 1, a, 1, a, 1])  
    circle.knots = np.array([0,0,0,0.5,0.5,1.0,1.0,1.5,1.5,2,2,2])*np.pi
    circle.N = len(circle.w)
    circle.dims = 3      # physical dimesions
    circle.p = 2         # NURBS basis degree

    # visualize it
    circle.View()
 
    return

# call main
if __name__ == '__main__':
    main()
