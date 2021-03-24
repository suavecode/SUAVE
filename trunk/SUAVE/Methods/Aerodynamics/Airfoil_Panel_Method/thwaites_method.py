## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
# thwaites_method.py 

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
# thwaites_method
# ----------------------------------------------------------------------  
## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def thwaites_method(theta_0, L, Re_L, x_i, Ve_i, dVe_i):
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
    nu         = L / Re_L
    n          = 100
    y0         = theta_0**2 * getVe(0,x_i,Ve_i)**6
    xspan      = np.linspace(0,L,n) 
    theta2_Ve6 = odeint(odefcn, y0, xspan, args=(nu, x_i, Ve_i)) 
    x          = np.zeros(n)  
    H          = np.zeros_like(x)
    cf         = np.zeros_like(x)
    del_star   = np.zeros_like(x)
    theta      = np.zeros_like(x)
    Re_theta   = np.zeros_like(x)
    Re_x       = np.zeros_like(x)
    
    for i in range(n): 
        theta[i]    = np.sqrt(theta2_Ve6[i]/ getVe(x[i], x_i, Ve_i)**6)
        lambda_val  = theta[i]**2 * getdVe(x[i],x_i,dVe_i) / nu
        H[i]        = getH(lambda_val )
        Re_theta[i] = getVe(x[i],x_i,Ve_i) * theta[i] / nu
        Re_x[i]     = getVe(x[i],x_i,Ve_i) * x[i] / nu
        cf[i]       = getcf(lambda_val ,Re_theta[i])
        del_star[i] = H[i]*theta[i]
        
    return x, theta, del_star, H, cf, Re_theta, Re_x  
    
def getH(lambda_val ):
    if lambda_val  > 0.0:
        H = 2.61 - 3.75*lambda_val  + 5.24*lambda_val **2
    else:
        H = 0.0731 / (0.14 + lambda_val ) + 2.088
    return H 
    
def odefcn(y,x, nu,x_i,Ve_i):
    dydx = 0.45*getVe(x,x_i,Ve_i)**5*nu
    return dydx 
    
def getVe(x,x_i,Ve_i):
    Ve_func = interp1d(x_i,Ve_i,fill_value = "extrapolate")
    Ve = Ve_func(x)
    return Ve 

def getdVe(x,x_i,dVe_i):
    dVe_func = interp1d(x_i,dVe_i,fill_value = "extrapolate")
    dVe = dVe_func(x)
    return dVe 

def  getcf(lambda_val , Re_theta):
    if lambda_val  > 0.0:
        l = 0.22 + 1.57 * lambda_val  - 1.8 * lambda_val **2
    else:
        l = 0.22 + 1.402 * lambda_val  + 0.018 * lambda_val  / (0.107 + lambda_val ) 
    cf = 2*l/Re_theta 
    
    return cf 