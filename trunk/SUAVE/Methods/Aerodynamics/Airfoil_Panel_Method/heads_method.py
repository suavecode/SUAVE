## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
# heads_method.py 
# Created:  Mar 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data 
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import odeint  
# ----------------------------------------------------------------------
# heads_method.py 
# ----------------------------------------------------------------------   
## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def heads_method(npanel,nalpha,nRe,DEL_0,THETA_0,DELTA_STAR_0, TURBULENT_SURF,RE_L,TURBULENT_COORD,
                 VE_I, DVE_I,batch_analysis,tol= 1E0):
    """ Computes the boundary layer characteristics in turbulent
    flow pressure gradients

    Source:
    Head, M. R., and P. Bandyopadhyay. "New aspects of turbulent boundary-layer structure."
    Journal of fluid mechanics 107 (1981): 297-338.

    Assumptions:
    None  

    Inputs: 
    nalpha         - number of angle of attacks                                                    [unitless]
    nRe            - number of reynolds numbers                                                    [unitless]
    batch_analysis - flag for batch analysis                                                       [boolean]
    DEL_0          - intital bounday layer thickness                                               [m]
    DELTA_STAR_0   - initial displacement thickness                                                [m]
    THETA_0        - initial momentum thickness                                                    [m]
    TURBULENT_SURF - normalized length of surface                                                  [unitless]
    RE_L           - Reynolds number                                                               [unitless]
    TURBULENT_COORD- x coordinate on surface of airfoil                                            [unitless] 
    VE_I           - boundary layer velocity at transition location                                [m/s-m] 
    DVE_I          - intial derivative value of boundary layer velocity at transition location     [unitless] 
    npanel         - number of points on surface                                                   [unitless]
    tol            - boundary layer error correction tolerance                                     [unitless]

    Outputs: 
    RESULTS.
      X_H          - reshaped distance along airfoil surface                    [unitless]
      THETA_H      - momentum thickness                                         [m]
      DELTA_STAR_H - displacement thickness                                     [m] 
      H_H          - shape factor                                               [unitless]
      CF_H         - friction coefficient                                       [unitless]
      RE_THETA_H   - Reynolds number as a function of momentum thickness        [unitless]
      RE_X_H       - Reynolds number as a function of distance                  [unitless]
      DELTA_H      - boundary layer thickness                                   [m]
       
    Properties Used:
    N/A
    """   
     
    # Initialize vectors 
    X_H          = np.zeros((npanel,nalpha,nRe))
    THETA_H      = np.zeros_like(X_H)
    DELTA_STAR_H = np.zeros_like(X_H)
    H_H          = np.zeros_like(X_H)
    CF_H         = np.zeros_like(X_H) 
    RE_THETA_H   = np.zeros_like(X_H)
    RE_X_H       = np.zeros_like(X_H)
    DELTA_H      = np.zeros_like(X_H)       

    if batch_analysis:
        N_ALPHA = nalpha
    else:
        N_ALPHA = 1  
    for a_i in range(N_ALPHA):
        for re_i in range(nRe):   
            if not batch_analysis:  
                a_i = re_i  
            # length of tubulent surface  
            l = TURBULENT_SURF[a_i,re_i] 
            if l == 0.0:
                pass
            else: 
                theta_0      = THETA_0[a_i,re_i] 
                Re_L         = RE_L[a_i,re_i] 
                nu           = l/Re_L    
                x_i          = TURBULENT_COORD.data[:,a_i,re_i][TURBULENT_COORD.mask[:,a_i,re_i] ==False] 
                Ve_i         = VE_I.data[:,a_i,re_i][TURBULENT_COORD.mask[:,a_i,re_i] ==False]
                dVe_i        = DVE_I.data[:,a_i,re_i][TURBULENT_COORD.mask[:,a_i,re_i] ==False]  
                del_0        = DEL_0[a_i,re_i]
                del_star_0   = DELTA_STAR_0[a_i,re_i]
                H_0          = del_star_0 / theta_0
                H1_0         = getH1(np.atleast_1d(H_0))[0]
                if np.isnan(H1_0):
                    H1_0     = (del_0 - del_star_0) / theta_0 
                y0           = [theta_0, getVe(0,x_i,Ve_i)*theta_0*H1_0]     
                y            = odeint(odefcn,y0,x_i,args=(Re_L/l, x_i, Ve_i, dVe_i))  
                
                # Compute momentum thickness, theta 
                theta        = y[:,0] 
                Ve_theta_H1  = y[:,1]    
                
                # find theta values that do not converge and replace them with neighbor
                idx1         = np.where(abs((theta[1:] - theta[:-1])/theta[:-1]) > tol )[0]
                if len(idx1)> 1: 
                    np.put(theta,idx1 + 1, theta[idx1])    
                idx1         = np.where(abs((Ve_theta_H1[1:] - Ve_theta_H1[:-1])/Ve_theta_H1[:-1]) >tol)[0]
                if len(idx1)> 1: 
                    np.put(Ve_theta_H1,idx1 + 1, Ve_theta_H1[idx1])    
                  
                # Compute mass flow shape factor, H1
                H1           = Ve_theta_H1/(theta*Ve_i)
                
                # Compute H 
                H            = getH(np.atleast_1d(H1)) 
                H[H<0]       = 1E-6    # H cannot be negative 
                # find H values that do not converge and replace them with neighbor
                idx1               = np.where(abs((H[1:] - H[:-1])/H[:-1]) >tol)[0]
                if len(idx1)> 1: 
                    np.put(H,idx1 + 1, H[idx1])     
                
                # Compute Reynolds numbers based on momentum thickness  
                Re_theta     = Re_L/l * Ve_i*theta 
                
                # Compute Reynolds numbers based on distance along airfoil
                Re_x         = Ve_i* x_i / nu
                
                # Compute skin friction 
                cf           = getcf(np.atleast_1d(Re_theta),np.atleast_1d(H))
                cf[cf<0]     = 1E-6 
                
                # Compute displacement thickness
                del_star     = H*theta   
                
                # Compute boundary layer thickness 
                delta        = theta*H1 + del_star 
                delta[0]     = 0  
                
                # Reynolds number at x=0 cannot be negative (give nans)
                Re_x[0]      = 1E-5                
    
                # Find where matrices are not masked 
                indices = np.where(TURBULENT_COORD.mask[:,a_i,re_i] == False)
                
                # Store results 
                np.put(X_H[:,a_i,re_i],indices,x_i )
                np.put(THETA_H[:,a_i,re_i],indices,theta)
                np.put(DELTA_STAR_H[:,a_i,re_i],indices,del_star)
                np.put(H_H[:,a_i,re_i],indices,H)
                np.put(CF_H[:,a_i,re_i],indices ,cf)
                np.put(RE_THETA_H[:,a_i,re_i],indices,Re_theta)
                np.put(RE_X_H[:,a_i,re_i],indices,Re_x)
                np.put(DELTA_H[:,a_i,re_i],indices,delta) 

    RESULTS = Data(
        X_H          = X_H,      
        THETA_H      = THETA_H,   
        DELTA_STAR_H = DELTA_STAR_H,
        H_H          = H_H,       
        CF_H         = CF_H,   
        RE_THETA_H   = RE_THETA_H,   
        RE_X_H       = RE_X_H,    
        DELTA_H      = DELTA_H,   
    )    

    return  RESULTS

def getH(H1):
    """ Computes the shape factor, H
    Assumptions:
    None
    Source:
    None
    Inputs: 
    H1       - mass flow shape factor [unitless]
    Outputs:  
    H        - shape factor [unitless]
    Properties Used:
    N/A
    """         
    H       = 0.6778 + 1.1536*(H1-3.3)**-0.326
    idx1    = (H1 < 3.3)
    H[idx1] = 3.0
    idx2    = (H1 > 5.3)
    H[idx2] = 1.1 + 0.86*(H1[idx2] - 3.3)**-0.777 
    return H 

def getH1(H) :    
    """ Computes the mass flow shape factor, H1
    Assumptions:
    None
    Source:
    None
    Inputs: 
    H        - shape factor [unitless]
    Outputs:  
    H1       - mass flow shape factor [unitless]
    Properties Used:
    N/A 
    """
    H1       = 3.3 + 0.8234*(H - 1.1)**-1.287  
    idx1     = (H > 1.6) 
    H1[idx1] = 3.3 + 1.5501*(H[idx1] - 0.6778)**-3.064
    return H1 

def odefcn(y,x,ReL_div_L, x_i, Ve_i, dVe_i): 
    """ Computes boundary layer functions using SciPy ODE solver 
    Assumptions:
    None
    Source:
    None
    Inputs:  
    y           - initial conditions of functions               [unitless]
    x           - new x values at which to solve ODE            [unitless]
    ReL_div_L   - ratio of Reynolds number to length of surface [unitless]
    x_i         - intial array of x values                      [unitless]
    Ve_i        - intial boundary layer velocity                [m/s]
    dVe_i       - initial derivative of bounday layer velocity  [m/s-m]

    Outputs:  
    f           - 2D function of momentum thickness and the product of 
                  the velocity,momentum thickness and the mass flow shape factor
    Properties Used:
    N/A 
    """    
    theta       = y[0]
    Ve_theta_H1 = y[1]  

    if theta == 0:
        H1 = Ve_theta_H1 / (theta + 1e-6) / getVe(x,x_i,Ve_i)
    else:
        H1 = Ve_theta_H1 / theta / getVe(x,x_i,Ve_i)

    H           = getH(np.atleast_1d(H1))
    Re_theta    = ReL_div_L * theta
    cf          = getcf(np.atleast_1d(Re_theta),np.atleast_1d(H))
    dydx_1      = 0.5*cf-(theta/getVe(x,x_i,Ve_i))*(2+H)*getdVe(x, x_i, dVe_i)
    dydx_2      = getVe(x,x_i,Ve_i)*0.0306*(H1 - 3)**-0.6169 
    f           = [dydx_1,dydx_2] 
    return f 

def getVe(x,x_i,Ve_i):
    """ Interpolates the bounday layer velocity over a new dimension of x 
    Assumptions:
    None
    Source:
    None
    Inputs: 
    x         - new x dimension                    [unitless]
    x_i       - old x dimension                    [unitless]
    Ve_i      - old boundary layer velocity values [m/s] 

    Outputs:  
    Ve        - new boundary layer velocity values [m/s]
    Properties Used:
    N/A 
    """    
    Ve_func = interp1d(x_i,Ve_i,fill_value = "extrapolate")
    Ve      = Ve_func(x)
    return Ve 

def getdVe(x,x_i,dVe_i):
    """ Interpolates the derivatives of the bounday layer velocity over a new dimension of x

    Assumptions:
    None
    Source:
    None
    Inputs: 
    x         - new x dimension                                  [unitless]
    x_i       - old x dimension                                  [unitless]
    dVe_i     - old derivative of boundary layer velocity values [m/s-m] 

    Outputs:  
    dVe       - new derivative of boundary layer velocity values [m/s-m]
    Properties Used:
    N/A 
    """        
    dVe_func = interp1d(x_i,dVe_i,fill_value = "extrapolate")
    dVe      = dVe_func(x)
    return dVe  

def getcf(Re_theta,H): 
    """ Computes the skin friction coefficient, cf
    Assumptions:
    None
    Source:
    None
    Inputs: 
    Re_theta - Reynolds Number as a function of momentum thickness [m]
    H        - shape factor                                        [unitless]
    Outputs:  
    cf       - skin friction coefficient  [unitless]
    Properties Used:
    N/A 
    """    
    cf       = 0.246*10**(-0.678*H)*(Re_theta)**-0.268 
    idx1     = (Re_theta == 0) 
    cf[idx1] = 0.246*10**(-0.678*H[idx1])*(1e-3)**-0.268 
    return cf 