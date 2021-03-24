## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
# airfoil_analysis.py

# Created:  Mar 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt   


from .hess_smith      import hess_smith
from .thwaites_method import thwaites_method
from .heads_method    import heads_method
from .aero_coeff      import aero_coeff

# ----------------------------------------------------------------------
# panel_geometry.py
# ----------------------------------------------------------------------   

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def airfoil_analysis(x_coord,y_coord,alpha,npanel = 100):
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
    # Begin by solving for velocity distribution at airfoil surface ucosg  inviscid panel simulation 
    x,y,vt,cos_t = hess_smith(x_coord,y_coord,alpha,npanel)
    
    # Find stagnation point
    for i in range(npanel):
        if vt[i] > 0:
            i_stag = i   # represents last index on bottom surface
            break 
    
    # flip arrays on bottom surface and re-parameterise x to represent arc length
    # measured from stagnation point on bottom surface
    x_bot_vals         =  x[:i_stag][::-1]
    cos_t_bot     = -cos_t [:i_stag][::-1]
    x_bot    = np.zeros_like(x_bot_vals ) 
    y_bot         =  y[:i_stag][::-1]
    for i in range(1,len(x_bot_vals )):
        x_bot[1:] = x_bot[:-1] + np.sqrt((x_bot_vals[1:] - x_bot_vals [:-1])**2 + (y_bot[1:] - y_bot[:-1])**2)
     
    Ve_bot       = -vt[:i_stag][::-1] # negative because the velocity is measured anti-clockwise around surface
    #cp_bot       =    
    dVe_bot      = np.zeros_like(x_bot_vals)
    dVe_bot_temp = np.diff(Ve_bot)/np.diff(x_bot_vals)
    dVe_bot[0]   = dVe_bot_temp[0]
    
    for i in range(1,len(x_bot)- 1 ):
        a = x_bot[i] - x_bot[i-1]
        b = x_bot[i+1] - x_bot[i-1]
        dVe_bot[i] = (b*dVe_bot_temp[i-1] + a*dVe_bot_temp[i])/(a+b)
    
    dVe_bot[-1] = dVe_bot_temp[-1]
    
    #re-parameterise based on length of boundary for the top surface of the airfoil
    x_top_vals      = x[i_stag:]
    cos_t_top  = cos_t[i_stag:]
    x_top = np.zeros_like(x_top_vals)
    y_top      = y[i_stag:]
    
    for i in range(1,len(x_top_vals)):
        x_top[i] = x_top[i-1] + np.sqrt((x_top_vals[i] - x_top_vals[i-1])**2 + (y_top[i] - y_top[i-1])**2)
     
    Ve_top       = vt[i_stag:]
    #cp_top       = 1 - (Uu/Uinf)**2 
    dVe_top      = np.zeros_like(x_top_vals)
    dVe_top_temp = np.diff(Ve_top)/np.diff(x_top)
    dVe_top[0]   = dVe_top_temp[0]
    for  i in range(1,len(x_top)- 1):
        a          = x_top[i] - x_top[i-1]
        b          = x_top[i+1] - x_top[i-1]
        dVe_top[i] = (b*dVe_top_temp[i-1] + a*dVe_top_temp[i])/(a+b)
    
    dVe_top[-1] = dVe_top_temp[-1] 
    
    #Plot velocity distributions
    fig  = plt.figure(10)
    axis = fig.add_subplot(1,1,1)     
    axis.plot(x_top,Ve_top,'-',x_bot,Ve_bot,'--')
    axis.set_xlabel('Distance from stagnation point')
    axis.set_ylabel('V_e') 
    axis.set_xlim([0,1.1])
    axis.set_ylim([0,1.5])        
    
    fig  = plt.figure(11)
    axis = fig.add_subplot(1,1,1)     
    axis.plot(x_top,dVe_top,'-',x_bot,dVe_bot,'--')
    axis.set_xlabel('Distance from stagnation point')
    axis.set_ylabel('dV_e/dx') 
    axis.set_xlim([0,1.1])
    axis.set_ylim([-5, 5])      
 
 
    # bottom
    Re_L           = 5E6
    L              = x_bot[-1]
    dVe_0_func     = interp1d(x_bot,dVe_bot,fill_value = "extrapolate") 
    dVe_0          = dVe_0_func (0)   
    theta_0        = np.sqrt(0.075*L/dVe_0/Re_L)
    x_t, theta_t, del_star_t, H_t, cf_t, Re_theta_t, Re_x_t= thwaites_method(0.000001, L, Re_L, x_bot, Ve_bot, dVe_bot)
    tr_crit        = Re_theta_t - 1.174*(1 + 224000/Re_x_t)*Re_x_t**0.46
    
    fig  = plt.figure(100)
    axis = fig.add_subplot(1,1,1)   
    axis.plot(x_t,Re_theta_t,'-', label = '$Re_\theta$')
    axis.plot(x_t,Re_theta_t-tr_crit,'--', label = '1.174(1+22400/Re_x)Re_x**{0.46}')
    axis.legend(loc='upper right')   
    axis.set_title('Transition Bottom Surface') 
    
    for i in range(len(tr_crit)):
        if tr_crit[i] > 0:
            interp_factor = tr_crit[i-1]/(tr_crit[i] - tr_crit[i-1])
            x_tr          = interp_factor*(x_t[i] - x_t[i-1]) + x_t[i-1]
            del_star_tr   = interp_factor*(del_star_t[i] - del_star_t[i-1]) + del_star_t[i-1]
            theta_tr      = interp_factor*(theta_t[i] - theta_t[i-1]) + theta_t[i-1]
            i_tr          = i
            break  
    
    x_h, theta_h, del_star_h, H_h, cf_h = heads_method(theta_tr, del_star_tr, L - x_tr, Re_L, x_bot - x_tr, Ve_bot, dVe_bot)
    x              = np.concatenate( [x_t[:i_tr], x_h + x_tr] )
    theta          = np.concatenate( [theta_t[:i_tr] ,theta_h] )
    del_star       = np.concatenate( [del_star_t[:i_tr] ,del_star_h] )
    H              = np.concatenate( [H_t[:i_tr] ,H_h] )
    cf             = np.concatenate( [cf_t[:i_tr], cf_h] )
    cos_t_bot_func = interp1d(x_bot,cos_t_bot,fill_value = "extrapolate")   
    cd             = cf*cos_t_bot_func(x)
    CD_top         = np.trapz(cd,x)
    
    fig  = plt.figure(2)
    axis = fig.add_subplot(1,1,1)   
    axis.plot(x, theta,'b-', label = '$\theta$' )
    axis.plot(x,del_star,'b--', label = '$\delta$*')
    axis.set_xlabel('x')
    axis.set_ylabel('Thickness') 
    axis.legend(loc='upper right')   
    fig  = plt.figure(3)
    axis = fig.add_subplot(1,1,1)   
    axis.plot(x, H)
    axis.set_xlabel('x')
    axis.set_ylabel('H') 
    
    fig  = plt.figure(4)
    axis = fig.add_subplot(1,1,1)   
    axis.plot(x, cf)
    axis.set_xlabel('x')
    axis.set_ylabel('c_f') 
    axis.set_xlim([0,1,])
    axis.set_ylim([0,0.005])      
    
    #Compute Top Surface
    L              = x_top[-1] 
    dVe_0_func     = interp1d(x_top,dVe_top,fill_value = "extrapolate")  
    dVe_0          = dVe_0_func(0)
    theta_0        = np.sqrt(0.075*L/dVe_0/Re_L)
    x_t, theta_t, del_star_t, H_t, cf_t, Re_theta_t, Re_x_t = thwaites_method(0.000001,L, Re_L, x_top, Ve_top, dVe_top) 
    tr_crit        = Re_theta_t - 1.174*(1 + 224000/Re_x_t)*Re_x_t**0.46
    
    fig  = plt.figure(110)
    axis = fig.add_subplot(1,1,1)   
    axis.plot(x_t,Re_theta_t,'-', label = '$Re_\theta$')
    axis.plot(x_t,Re_theta_t-tr_crit,'--', label = '1.174(1+22400/Re_x)Re_x**{0.46}') 
    axis.legend(loc='upper right')   
    axis.set_title('Transition Top Surface') 
    
    for i in range( len(tr_crit)):
        if tr_crit[i] > 0:
            interp_factor = tr_crit[i-1]/(tr_crit[i] - tr_crit[i-1])
            x_tr          = interp_factor*(x_t[i] - x_t[i-1]) + x_t[i-1]
            del_star_tr   = interp_factor*(del_star_t[i] - del_star_t[i-1]) + del_star_t[i-1]
            theta_tr      = interp_factor*(theta_t[i] - theta_t[i-1]) + theta_t[i-1]
            i_tr          = i
            break 
    
    x_h, theta_h, del_star_h, H_h, cf_h = heads_method(theta_tr, del_star_tr, L - x_tr, Re_L, x_top - x_tr, Ve_top, dVe_top)
    x               = np.concatenate( [x_t[:i_tr], x_h + x_tr] )
    theta           = np.concatenate( [theta_t[:i_tr] ,theta_h] )
    del_star        = np.concatenate( [del_star_t[:i_tr], del_star_h] )
    H               = np.concatenate( [H_t[:i_tr] ,H_h] )
    cf              = np.concatenate( [cf_t[:i_tr], cf_h] )
    cos_t_top_func  = interp1d(x_top,cos_t_top,fill_value = "extrapolate") 
    cd              = cf*cos_t_top_func(x)   
    CD_bottom       = np.trapz(cd,x)
    CD_total        = CD_bottom + CD_top
    
    fig  = plt.figure(6)
    axis = fig.add_subplot(1,1,1)   
    axis.plot(x, theta,'b-',label = '$\theta$')
    axis.plot(x,del_star,'b--' , label = '$\delta*$')
    axis.set_xlabel('x')
    axis.set_ylabel('Thickness') 
    axis.legend(loc='upper right')   
     
    fig  = plt.figure(7)
    axis = fig.add_subplot(1,1,1) 
    axis.plot(x, H)
    axis.set_xlabel('x')
    axis.set_ylabel('H') 
    
    fig  = plt.figure(8)
    axis = fig.add_subplot(1,1,1) 
    axis.plot(x, cf)
    axis.set_xlabel('x')
    axis.set_ylabel('c_f') 
    axis.set_xlim([0,1.4])
    axis.set_ylim([0,0.005])    
    
    #cp = np.concatenate([cp_top,cp_bot])
    #cl,cd,cm = aero_coeff(x,y,cp,alpha,npanel)  
    
    return  
