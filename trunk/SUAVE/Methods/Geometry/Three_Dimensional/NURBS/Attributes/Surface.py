# NURBS.py
#
# Created By:       M. Colonno  6/27/13
# Updated:          M. Colonno  7/8/13

""" SUAVE NURBS containers """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
import numpy as np


# ----------------------------------------------------------------------
#  Surface class
# ----------------------------------------------------------------------

class Surface(Data):

    """ SUAVE NURBS surface class """

    def __defaults__(self):         
        self.uknots = []  
        self.vknots = []
        self.p = 3
        self.q = 3
        self.N = 0
        self.M = 0
        self.CPs = Data()
        self.CPs.x = []; self.CPs.y = []; self.CPs.z = []; self.w = []
        
    def Build(self,Curves,degree = 3,knots = []):   
        
        """  .Build(Curves,knots,degree): determine the know span index
    
         Inputs:    Curves = list of NURBSCurve class instances     (required)      (list of Curves)
                    knots = array of knot coordinates in v          (optional)      (array of floats)
                    degree = degree of curves in v-direction        (default = 3)   (int)

         Outputs:   Surface class instance, populated 

        """
       
        # error checking 
        self.uknots = Curves[0].knots  
        for curve in Curves:
            if len(curve.knots) != len(self.uknots):
                print "Knots vectors of curves have different lengths, no surface created"
                return 

        self.N = len(Curves[0].CPs.x)
        self.M = len(Curves)

        self.p = Curves[0].p
        self.q = degree

        # knot vector in v
        if not knots:

            # create knot vector 0 ---> 1 (FIX: non-uniform spacing - MC)
            self.vknots = np.zeros(degree)
            self.vknots = np.append(self.vknots,np.linspace(0,1,self.M-degree+1))
            self.vknots = np.append(self.vknots,np.ones(degree))
     
        else:
            self.vknots = vknots
        
        # weights
        self.uw = Curves[0].w
        self.vw = np.ones(self.N)

        # build control point and weight net
        self.CPs.x = np.zeros((self.N,self.M))
        self.CPs.y = np.zeros((self.N,self.M))
        self.CPs.z = np.zeros((self.N,self.M))
        self.w = np.zeros((self.N,self.M))
        for i in range(self.N):
            for j in range(self.M):
                self.CPs.x[i][j] = Curves[j].CPs.x[i]
                self.CPs.y[i][j] = Curves[j].CPs.y[i]
                self.CPs.z[i][j] = Curves[j].CPs.z[i]
                self.w[i][j] = Curves[j].w[i]

    def Evaluate(self,u,v):

        """  .Evaluate(u,v): evaluate a surface at (u,v)
    
        Inputs:     u = parametric coordinate 1             (required)      (float)
                    v = parametric coordinate 2             (required)      (float)

         Outputs:   coordinates on surface at (u,v)     (list of floats)
    
        """

        # package data
        uCurve = Curve()
        uCurve.p = self.p
        uCurve.knots = self.uknots

        vCurve = Curve()
        vCurve.p = self.q
        vCurve.knots = self.vknots

        # find patch
        uspan = uCurve.FindSpan(u)
        vspan = vCurve.FindSpan(v)

        # compute basis functions
        Nu = uCurve.BasisFunctions(u,uspan)
        Nv = vCurve.BasisFunctions(v,vspan)
    
        # compute coordinates
        S = np.zeros(3); W = 0.0
        for l in range(0,self.q+1):                       # for (l = 0; l <= q; l++)
            t = np.zeros(3); w = 0.0
            vi = vspan - self.q + l   
            for k in range(0,self.p+1):                   # for (k = 0; k <= p; k++)
                ui = uspan - uCurve.p + k
                t[0] += Nu[k]*self.CPs.x[ui][vi]*self.w[ui][vi]
                t[1] += Nu[k]*self.CPs.y[ui][vi]*self.w[ui][vi]
                t[2] += Nu[k]*self.CPs.z[ui][vi]*self.w[ui][vi]
                w += Nu[k]*self.w[ui][vi]

            S += Nv[l]*t
            W += Nv[l]*w

        return S/W

    def View(self,Nu = 41,Nv = 41,show_CPs = True):

        """  .View(Nu,Nv,show_CPs): visualize a surface
    
         Inputs:    Nu = number of points to show in u          (optional, default = 41)    (int)
                    Nv = number of points to show in v          (optional, default = 41)    (int)
                    show_CPs = flag to show control point mesh  (optional, default = True)  (bool)

         Outputs:   3D mesh plot of surface (matplotlib figure)

        """

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # generate mesh points
        u = np.linspace(0,self.uknots[-1],Nu)
        v = np.linspace(0,self.vknots[-1],Nv)
        x = np.zeros((Nu,Nv)); y = np.zeros((Nu,Nv)); z = np.zeros((Nu,Nv))

        for i in range(Nu):
            for j in range(Nv):
                f = self.Evaluate(u[i],v[j])
                x[i][j] = f[0]; y[i][j] = f[1]; z[i][j] = f[2];

        # visualize
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # plot mesh
        ax.plot_wireframe(x,y,z,rstride=1,cstride=1)

        # plot control points
        if show_CPs:
            for i in range(self.N):
                ax.plot(self.CPs.x[i,:],self.CPs.y[i,:],self.CPs.z[i,:],'o--')
            for j in range(self.M):
                ax.plot(self.CPs.x[:,j],self.CPs.y[:,j],self.CPs.z[:,j],'o--')

        # labels, etc.
        plt.xlabel('x'); plt.ylabel('y'); plt.axis("equal"); # plt.zlabel('z'); 
        if show_CPs:
            plt.title('NURBS Surface with Control Points')
        else:
            plt.title('NURBS Surface')
        plt.grid(True)
        plt.show()

        return 