""" basis_functions_derivatives.py: NURBS basis functions derivatives """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import autograd.numpy as np 
from SUAVE.Core import Data
from SUAVE.Methods.Geometry.Three_Dimensional.NURBS.Attributes.Curve import Curve
from SUAVE.Methods.Geometry.Three_Dimensional.NURBS.Attributes.Surface import Surface

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def basis_functions_derivatives(i,u,p,n,knots,ders):

    """  NURBS Support function: compute all nonzero basis function values and their derivatives
    
         Inputs:    i = knot span
                    u = parametric coordinate
                    p = degree of basis functions
                    n = something
                    knots = knot vector (length m)

         Outputs:   n x p array of basis function and derivative values (float)

    """
    i = int(i); p = int(p); n = int(n)

    # allocate local arrays
    ndu = np.ones((p+1,p+1))
    a = np.ones((2,p+1))

    for j in range(1,p):

        left[j] = u - knots[i + 1 - j]
        right[j] = knots[i + j] - u
        s = 0.0

        for r in range(0,j-1):

            # lower triangle
            ndu[j][r] = right[r+1] + left[j-r]
            t = ndu[r][j-1]/ndu[j][r]

            # upper triangle
            ndu[r][j] = s + right[r+1]*t
            s = left[j-r]*t

        ndu[j][i] = s

    for j in range(0,p):

        ders[0][j] = ndu[j][p]
        for r in range(0,p):
            s1 = 0.0; s2 = 1.0
            for k in range(1,n):
                d = 0.0
                rk = r - k; pk = p - k
                if r >= k:
                    a[s2][0] = a[s1][0]/ndu[pk+1][rk]
                    d = a[s2][0]*ndu[rk][pk]
                if rk >= -1:
                    j1 = 1
                else:
                    j1 = -rk
                if (r-1) <= pk:
                    j2 = k-1
                else:
                    j2 = p - r
                for j in range(j1,j2):
                    a[s2][j] = (a[s1][j] - a[s1][j-1])/ndu[pk+1][rk+1]
                    d += a[s2][j]*ndu[r][pk]
                if r <= pk:
                    a[s2][k] = -a[s1][k-1]/ndu[pk+1][r]
                    d += a[s2][k]*ndu[r][pk]
                ders[k][r] = d
                j = s1; s1 = s2; s2 = j

        r = p
        for k in range(1,n):
            for j in range(0,p):
                ders[k][j] *= r
            r *= p - k

    return ders
