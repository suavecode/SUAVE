## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
# heads_method.py 
# Created:  Mar 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import odeint
# ----------------------------------------------------------------------
# heads_method.py 
# ----------------------------------------------------------------------   
## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def heads_method(theta_0, del_star_0, L, Re_L, x_i, Ve_i, dVe_i):
    """

    Assumptions:
    None

    Source:
    None

    Inputs: 

    Outputs: 

    Properties Used:
    N/A
    """        
    n            = 100
    H_0          = del_star_0 / theta_0
    y0           = [theta_0, getVe(0,x_i,Ve_i)*theta_0*getH1(H_0)]
    xspan        = np.linspace(0,L,n)  
    ReL_div_L    = Re_L/L
    y            = odeint(odefcn,y0,xspan,args=(ReL_div_L, x_i, Ve_i, dVe_i)) 
    theta        = y[:,0] 
    Ve_theta_H1  = y[:,1]
    x            =  np.linspace(0,L,n) 
    H1           = np.zeros_like(x)
    H            = np.zeros_like(x)
    cf           = np.zeros_like(x)
    del_star     = np.zeros_like(x)

    for i in range(n):
        H1[i]       = Ve_theta_H1[i]/theta[i]/getVe(x[i], x_i, Ve_i)
        H[i]        = getH(H1[i])
        Re_theta    = Re_L/L * getVe(x[i],x_i,Ve_i) * theta[i]
        cf[i]       = getcf(Re_theta,H[i])
        del_star[i] = H[i]*theta[i] 

    return x, theta, del_star, H, cf

def  getH(H1):
    if H1 < 3.3:
        H = 3.0
    elif H1 < 5.3:
        H = 0.6778 + 1.1536*(H1-3.3)**-0.326
    else:
        H = 1.1 + 0.86*(H1 - 3.3)**-0.777

    return H 

def  getH1(H) :   
    if H <= 1.6:
        H1 = 3.3 + 0.8234*(H - 1.1)**-1.287
    else:
        H1 = 3.3 + 1.5501*(H - 0.6778)**-3.064
    return H1


def odefcn(y,x,ReL_div_L, x_i, Ve_i, dVe_i): 
    theta       = y[0]
    Ve_theta_H1 = y[1]
    if theta == 0:
        H1 = Ve_theta_H1 / (theta + 1e-6) / getVe(x,x_i,Ve_i)
    else:
        H1 = Ve_theta_H1 / theta / getVe(x,x_i,Ve_i)

    H        = getH(H1)
    Re_theta = ReL_div_L * theta
    cf       = getcf(Re_theta,H)
    dydx_1   = 0.5*cf-theta/getVe(x,x_i,Ve_i)*(2+H)*getdVe(x, x_i, dVe_i)
    dydx_2   = getVe(x,x_i,Ve_i)*0.0306*(H1 - 3)**-0.6169

    f = [dydx_1,dydx_2]

    return f 

def getVe(x,x_i,Ve_i):
    Ve_func = interp1d(x_i,Ve_i,fill_value = "extrapolate")
    Ve = Ve_func(x)
    return Ve 

def getdVe(x,x_i,dVe_i):
    dVe_func = interp1d(x_i,dVe_i,fill_value = "extrapolate")
    dVe = dVe_func(x)
    return dVe  

def getcf(Re_theta,H):
    if Re_theta == 0:
        cf = 0.246*10**(-0.678*H)*(Re_theta + 1e-3)**-0.268
    else:
        cf = 0.246*10**(-0.678*H)*(Re_theta)**-0.268 

    return cf 