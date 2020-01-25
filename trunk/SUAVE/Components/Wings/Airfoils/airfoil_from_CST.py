## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil-CST
# create_airfoil_from_CST.py
# 
# Created:  Sep 2019, S.Karpuk
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data

import numpy as np


# ------------------------------------------------------------
#  creates airfoil points from a CST file
# ------------------------------------------------------------

## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil-CST

class airfoil_from_CST(Data):

    def __defaults__(self):
        """This sets the default values and methods for the analysis.
        Assumptions:
            Possible shape coefficients:
                                        N1 = 0.5    N2 = 1.0   - NACA type round nose and pointed aft end airfoil
                                        N1 = 0.5    N2 = 0.5   - elliptic airfoil
                                        N1 = 1.0    N2 = 1.0   - biconvex airfoil
        Source:
        None
        
        Inputs:
        None
        Outputs:
        None
        Properties Used:
        N/A
        """

        self.shape_coefficients    = Data()
        self.shape_coefficients.N1 = 0.5
        self.shape_coefficients.N2 = 1.0
        self.shape_coefficients.dz = 0.003                                          # trailing edge gap from the chord line

        self.points                  = Data()
        self.points.number_of_points = 40                                           # number of points for each half of the airfoil

        self.points.betta            = np.linspace(0, np.pi ,  \
                                                   num=self.points.number_of_points)

        self.points.x = np.zeros(len(self.points.betta))
        for i in range(len(self.points.betta)):
            self.points.x[i] = 0.5*(1-np.cos(self.points.betta[i]))

        
        
    def import_airfoil_dat(self,filename):
        """Creates airfoil points from a CST-file
        
        Assumptions:
        Airfoil file in Lednicer format
        Source:
        None                 
        Inputs:
        filename   <string>
        Outputs:
        data       numpy array with airfoil data
        Properties Used:
        N/A
        """     

        # Unpack
        dz = self.shape_coefficients.dz

        
        filein = open(filename,'r')
        header = filein.readline().strip()
        w1     = [float(x) for x in filein.readline().split()]
        w2     = [float(x) for x in filein.readline().split()]
        y_lower,x = self.create_CST_airfoil(w1,-dz)
        y_upper,x = self.create_CST_airfoil(w2,dz)
            
        return y_lower,y_upper,x


    def create_CST_airfoil(self,w,dz):
        """
        Runs a CST algorithm to compute the airfoil coordinates
                    
        Assumptions:
        Airfoil file in Lednicer format
        Source:
            B. Kulfan CST - Universal Parametric Geometry Representation Method
            With Applications to Supersonic Aircraft
            Fourth International Conference on Flow Dynamics Sendai International
            Center Sendai, Japan September 26-28, 2007
        Inputs:
        w         CST coordinates
        dz        trailing edge gap
        Outputs:
        y,x       numpy arrays with airfoil data
        Properties Used:
        N/A
        """  

        # Unpack
        N1 = self.shape_coefficients.N1
        N2 = self.shape_coefficients.N2
        x  = self.points.x
        print(self)
        
        
        # Class function; taking input of N1 and N2
        C = np.zeros(len(x))
        for i in range(len(x)):
            C[i] = x[i]**N1*((1-x[i])**N2)

        # Shape function; using Bernstein Polynomials
        n = len(w) - 1  # Order of Bernstein polynomials

        K = np.zeros(n+1)
        for i in range(0, n+1):
            K[i] = np.math.factorial(n)/(np.math.factorial(i)*(np.math.factorial((n)-(i))))

        S = np.zeros(len(x))
        for i in range(len(x)):
            S[i] = 0
            for j in range(0, n+1):
                S[i] += w[j]*K[j]*x[i]**(j) * ((1-x[i])**(n-(j)))

        # Calculate y output
        y = np.zeros(len(x))
        for i in range(len(y)):
            y[i] = C[i] * S[i] + x[i] * dz

        return y,x
