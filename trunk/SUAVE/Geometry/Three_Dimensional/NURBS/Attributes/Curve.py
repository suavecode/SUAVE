# NURBS.py
#
# Created By:       M. Colonno  6/27/13
# Updated:          M. Colonno  7/8/13

""" SUAVE NURBS containers """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data, Data_Exception
import numpy as np


# ----------------------------------------------------------------------
#  Curve class
# ----------------------------------------------------------------------

class Curve(Data):

    """ SUAVE NURBS curve class """

    def __defaults__(self):
        self.p = 3      # degree, e.g. cubic spline: order = 4, degree = 3
        self.N = 0
        self.dims = 2
        self.w = []
        self.knots = []
        self.CPs = Data()
        self.CPs.x = []; self.CPs.y = []; self.CPs.z = []

    def AddPoint(self,P,w=1.0):

        """  NURBSCurve.AddPoint(P,w=1.0): add a control point to a curve 
    
         Inputs:    P = control point to be added   (required)                  (list or array of floats)
                    w = corresponding weight        (optional, default = 1.0)   (float)

         Outputs:   point appended to list of control points for curve

        """

        # first point added, initialize arrays
        if self.N == 0:
            self.dims = len(P)
            self.CPs.x = np.array(P[0])
            self.CPs.y = np.array(P[1])
            if self.dims == 3:
                self.CPs.z = np.array(P[2])
            self.w = np.array(w)
        else:
            self.CPs.x = np.append(self.CPs.x,P[0])
            self.CPs.y = np.append(self.CPs.y,P[1])
            if self.dims == 3:
                self.CPs.z = np.append(self.CPs.z,P[2])
            self.w = np.append(self.w,w)

        self.N += 1

    def FindSpan(self,u):

        """  .FindSpan(u): determine the know span index of u
    
         Inputs:    u = parametric coordinate

         Outputs:   knot span index (int)

        """

        # unpack
        m = len(self.knots) 
        n = m - self.p - 1

        # special cases of endpoint
        if u == self.knots[-1]:
            return n - 1

        # binary search
        low = 0; high = m
        mid = (low + high)/2    # note: ints
        while u < self.knots[mid] or u >= self.knots[mid+1]:
            if u < self.knots[mid]:
                high = mid
            else:
                low = mid
            mid = (low + high)/2

        return mid

    def BasisFunctions(self,u,i):

        """  .BasisFunctions(u,i): compute all nonzero basis function values at u
    
         Inputs:    i = knot span
                    u = parametric coordinate
                    p = degree of basis functions
                    knots = knot vector (length m)

         Outputs:   length p+1 array of basis function values (float)

        """

        N = np.ones(self.p+1); left = np.zeros(self.p+1); right = np.zeros(self.p+1)

        for j in range(1,self.p+1):         # for (j = 1; j <= p; j++)

            left[j] = u - self.knots[i+1-j]
            right[j] = self.knots[i+j] - u
            s = 0.0

            for r in range(j):              # for (r = 0; r < j; r++)
                t = N[r]/(right[r+1] + left[j-r])
                N[r] = s + right[r+1]*t
                s = left[j-r]*t
            
            N[j] = s

        return N

    def Evaluate(self,u):

        """  NURBS Support function: compute the basis function N_i,p
    
         Inputs:    curve = NURBS curve class instance
                    u = parametric coordinate (float)

         Outputs:   coordinates on curve at u (numpy float array)

        """

        # find span
        span = self.FindSpan(u)

        # compute basis functions
        N = self.BasisFunctions(u,span)
    
        # compute coordinates
        C = np.zeros(self.dims); W = 0.0
        for i in range(0,self.p+1):            # for (i = 0; i <= p; i++)
            C[0] += N[i]*self.CPs.x[span-self.p+i]*self.w[span-self.p+i]
            C[1] += N[i]*self.CPs.y[span-self.p+i]*self.w[span-self.p+i]
            if self.dims == 3:
                C[2] += N[i]*self.CPs.z[span-self.p+i]*self.w[span-self.p+i]
            W += N[i]*self.w[span-self.p+i]

        return C/W

    def View(self,Nu = 101,show_CPs = True):

        """  .View(Nu,show_CPs): visualize a curve
    
         Inputs:    Nu = number of points to show in u          (optional, default = 101)    (int)
                    show_CPs = flag to show control point mesh  (optional, default = True)  (bool)

         Outputs:   3D plot of curve (matplotlib figure)

        """

        import matplotlib.pyplot as plt   

        # generate mesh points
        u = np.linspace(0,self.knots[-1],Nu)
        x = np.zeros(Nu); y = np.zeros(Nu); z = np.zeros(Nu)

        for i in range(Nu):
            f = self.Evaluate(u[i])
            x[i] = f[0]; y[i] = f[1]; z[i] = f[2];

        # visualize
        fig = plt.figure()
        if self.dims == 3:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.gca(projection='3d')
            ax.plot(x,y,z)
            if show_CPs:
                ax.plot(self.CPs.x,self.CPs.y,self.CPs.z,'o--')
            plt.xlabel('x'); plt.ylabel('y'); plt.axis("equal"); # plt.zlabel('z'); 
        elif self.dims == 2:
            plt.plot(x,y)
            if show_CPs:
                plt.plot(self.CPs.x,self.CPs.y,'o--')
            plt.xlabel('x'); plt.ylabel('y'); plt.axis("equal")

        if show_CPs:
            plt.title('Curve with Control Points')
        else:
            plt.title('Curve')

        plt.grid(True)
        plt.show()

        return 
