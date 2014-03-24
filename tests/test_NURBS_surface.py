# test_NURBS_surface.py: test various NURBS surface capability

import SUAVE
import numpy as np
from matplotlib.pyplot import  *
from mpl_toolkits.mplot3d import axes3d

# MAIN
def main():
    
    N = 8; p = 2;   # control points ; degree
    M = 9; q = 3;

    Curves = []
    a = np.sqrt(2)/2
    
    for i in range(N):      # loop over curves

        x = float(i)
        y = np.array([1, 1, 0, -1, -1, -1, 0, 1, 1]) + np.random.random()
        z = np.array([0, 1, 1, 1, 0, -1, -1, -1, 0]) + np.random.random()
        w = np.array([1, a, 1, a, 1, a, 1, a, 1])  # *np.sin(i*np.pi/(N-1))
        #x = np.linspace(0,1,M)
        #y = float(i)
        #z = np.random.random(M)
        
        curve = SUAVE.Attributes.Geometry3D.Curve()
        curve.p = p
        #curve.knots = np.zeros(p)
        #curve.knots = np.append(curve.knots,np.linspace(0,1,M-p+1))
        #curve.knots = np.append(curve.knots,np.ones(p))
        curve.knots = np.array([0,0,0,0.5,0.5,1.0,1.0,1.5,1.5,2,2,2])*np.pi
    
        for j in range(M):      # loop over CPs in a curve
            curve.AddPoint([x, y[j], z[j]], w[j])

        Curves.append(curve)

    surf = SUAVE.Attributes.Geometry3D.Surface()
    surf.Build(Curves,degree=q)
    surf.View()
 
    return

# call main
if __name__ == '__main__':
    main()
