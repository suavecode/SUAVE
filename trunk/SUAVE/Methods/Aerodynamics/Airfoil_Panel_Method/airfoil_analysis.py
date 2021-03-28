## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
# airfoil_analysis.py

# Created:  Mar 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import  Data
import numpy as np
from scipy.interpolate import interp1d 


from .hess_smith      import hess_smith
from .thwaites_method import thwaites_method
from .heads_method    import heads_method
from .aero_coeff      import aero_coeff

# ----------------------------------------------------------------------
# panel_geometry.py
# ----------------------------------------------------------------------   

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def airfoil_analysis(airfoil_geometry,alpha,Re_L,npanel = 100):
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
    
    x_coord = np.delete( airfoil_geometry.x_coordinates[0][::-1], int(npanel/2)) 
    y_coord = np.delete( airfoil_geometry.y_coordinates[0][::-1], int(npanel/2))   
    
    # Begin by solving for velocity distribution at airfoil surface ucosg  inviscid panel simulation 
    x,y,vt,cos_t,normals = hess_smith(x_coord,y_coord,alpha,npanel)    
    
    # Find stagnation point
    for i in range(npanel):
        if vt[i] > 0:
            i_stag = i   # represents last index on bottom surface
            break 

    # ---------------------------------------------------------------------
    # Bottom surface of airfoil 
    # ---------------------------------------------------------------------
    # x and y coordinates 
    x_bot_vals    =  x[:i_stag][::-1]  # flip arrays on bottom surface (measured from stagnation point on bottom surface)
    y_bot         =  y[:i_stag][::-1]  
    x_bot         = np.zeros_like(x_bot_vals )  
    x_bot[1:]     = np.cumsum(np.sqrt((x_bot_vals[1:] - x_bot_vals [:-1])**2 + (y_bot[1:] - y_bot[:-1])**2)) 
      
    # flow velocity and pressure of on botton surface 
    Ve_bot        = -vt[:i_stag][::-1] # negative because the velocity is measured anti-clockwise around surface 
    Cp_bot        = 1 - Ve_bot**2 
        
    # velocity gradients on bottom surface 
    dVe_bot       = np.zeros_like(x_bot_vals)
    dVe_bot_temp  = np.diff(Ve_bot)/np.diff(x_bot_vals)
    dVe_bot[0]    = dVe_bot_temp[0] 
    a             = x_bot[1:-2] - x_bot[:-3]
    b             = x_bot[2:-1] - x_bot[:-3]
    dVe_bot[1:-2] = (b*dVe_bot_temp[:-2] + a*dVe_bot_temp[1:-1])/(a+b) 
    dVe_bot[-1]   = dVe_bot_temp[-1]  
    L_bot         = x_bot[-1] # x - location of stagnation point 
    
    # laminar boundary layer properties using thwaites method 
    x_t_bot, theta_t_bot, del_star_t_bot, H_t_bot, cf_t_bot, Re_theta_t_bot, Re_x_t_bot,delta_t_bot= thwaites_method(0.000001, L_bot , Re_L, x_bot, Ve_bot, dVe_bot)
    
    # transition location  
    tr_crit_bot     = Re_theta_t_bot - 1.174*(1 + 224000/Re_x_t_bot)*Re_x_t_bot**0.46                
    tr_loc_vals     = np.where(tr_crit_bot > 0)[0] 
    if len(tr_loc_vals) == 0:# no trasition  
        i_tr_bot =  len(tr_crit_bot) - 1
    else: # transition 
        i_tr_bot = tr_loc_vals[0]    
   
    x_tr_bot        = x_t_bot[i_tr_bot] 
    del_star_tr_bot = del_star_t_bot[i_tr_bot] 
    theta_tr_bot    = theta_t_bot[i_tr_bot]    
    delta_tr_bot    = delta_t_bot[i_tr_bot] 
    
    x_h_bot, theta_h_bot, del_star_h_bot, H_h_bot, cf_h_bot, delta_h_bot = heads_method(delta_tr_bot,theta_tr_bot, del_star_tr_bot, L_bot - x_tr_bot, Re_L, x_bot - x_tr_bot, Ve_bot, dVe_bot)
    x_bs            = np.concatenate([x_t_bot[:i_tr_bot], x_h_bot + x_tr_bot] )
    theta_bot       = np.concatenate([theta_t_bot[:i_tr_bot] ,theta_h_bot] )
    del_star_bot    = np.concatenate([del_star_t_bot[:i_tr_bot] ,del_star_h_bot] )
    H_bot           = np.concatenate([H_t_bot[:i_tr_bot] ,H_h_bot] )
    cf_bot          = np.concatenate([cf_t_bot[:i_tr_bot], cf_h_bot] )
    delta_bot       = np.concatenate([delta_t_bot[:i_tr_bot], delta_h_bot]) 
    
    # ---------------------------------------------------------------------
    # Top surface of airfoil 
    # ---------------------------------------------------------------------
    
    #re-parameterise based on length of boundary for the top surface of the airfoil
    x_top_vals    = x[i_stag:] 
    y_top         = y[i_stag:] 
    x_top         = np.zeros_like(x_top_vals)
    x_top[1:]     = np.cumsum(np.sqrt((x_top_vals[1:] - x_top_vals[:-1])**2 + (y_top[1:] - y_top[:-1])**2))
    
    # flow velocity and pressure on top surface 
    Ve_top        = vt[i_stag:]  
    Cp_top        = 1 - Ve_top**2
    
    # velocity gradients on top surface 
    dVe_top       = np.zeros_like(x_top_vals)
    dVe_top_temp  = np.diff(Ve_top)/np.diff(x_top)
    dVe_top[0]    = dVe_top_temp[0]
    a             = x_top[1:-2] - x_top[:-3]
    b             = x_top[2:-1] - x_top[:-3]
    dVe_top[1:-2] = (b*dVe_top_temp[:-2] + a*dVe_top_temp[1:-1])/(a+b) 
    dVe_top[-1]   = dVe_top_temp[-1]   
    L_top         = x_top[-1]  
    
    # laminar boundary layer properties using thwaites method 
    x_t_top, theta_t_top, del_star_t_top, H_t_top, cf_t_top, Re_theta_t_top, Re_x_t_top, delta_t_top = thwaites_method(0.000001,L_top, Re_L, x_top, Ve_top, dVe_top) 
    
    # Mitchel's transition criteria (can often give nonsensical results) 
    tr_crit_top     = Re_theta_t_top - 1.174*(1 + 224000/Re_x_t_top)*Re_x_t_top**0.46     
    tr_loc_vals     = np.where(tr_crit_top > 0)[0] 
    if len(tr_loc_vals) == 0:  # manual trip point at stagnation point (fully turbulent)
        i_tr_top = 0
    else: # transition
        i_tr_top = tr_loc_vals[0]    
        
    x_tr_top        = x_t_top[i_tr_top] 
    del_star_tr_top = del_star_t_top[i_tr_top] 
    theta_tr_top    = theta_t_top[i_tr_top]    
    delta_tr_top    = delta_t_top[i_tr_top] 
    
    x_h_top, theta_h_top, del_star_h_top, H_h_top, cf_h_top, delta_h_top = heads_method(delta_tr_top,theta_tr_top, del_star_tr_top, L_top - x_tr_top, Re_L, x_top - x_tr_top, Ve_top, dVe_top)
    x_ts            = np.concatenate([x_t_top[:i_tr_top], x_h_top + x_tr_top])
    theta_top       = np.concatenate([theta_t_top[:i_tr_top] ,theta_h_top])
    del_star_top    = np.concatenate([del_star_t_top[:i_tr_top], del_star_h_top])
    H_top           = np.concatenate([H_t_top[:i_tr_top] ,H_h_top])
    cf_top          = np.concatenate([cf_t_top[:i_tr_top], cf_h_top])
    delta_top       = np.concatenate([delta_t_top[:i_tr_top], delta_h_top]) 
    
    # ---------------------------------------------------------------------
    # Concatenate Upper and Lower Surface Data 
    # ---------------------------------------------------------------------   

    x_vals                = x[::-1]  
    y_vals                = y[::-1]  
    Cp_vals               = np.concatenate([Cp_top[::-1],Cp_bot])
    Ve_vals               = np.concatenate([Ve_top[::-1],Ve_bot])
    dVe_vals              = np.concatenate([dVe_top[::-1],dVe_bot ])
                          
    theta_top_func        = interp1d(x_ts[::-1], theta_top[::-1])
    theta_bot_func        = interp1d(x_bs,theta_bot) 
    delta_star_top_func   = interp1d(x_ts[::-1] ,del_star_top[::-1])
    delta_star_bot_func   = interp1d(x_bs,del_star_bot) 
    H_top_func            = interp1d(x_ts[::-1],H_top[::-1])
    H_bot_func            = interp1d(x_bs,H_bot)   
    Cf_top_func           = interp1d(x_ts[::-1],cf_top[::-1])
    Cf_bot_func           = interp1d(x_bs,cf_bot) 
    delta_top_func        = interp1d(x_ts[::-1],delta_top[::-1])
    delta_bot_func        = interp1d(x_bs,delta_bot)     
                          
    theta_vals            = np.concatenate([theta_top_func(x_top[::-1]),theta_bot_func(x_bot)])
    delta_star_vals       = np.concatenate([delta_star_top_func(x_top[::-1]),delta_star_bot_func(x_bot)])
    H_vals                = np.concatenate([H_top_func(x_top[::-1]),H_bot_func(x_bot)])
    Cf_vals               = np.concatenate([Cf_top_func(x_top[::-1]),Cf_bot_func(x_bot)])
    delta_vals            = np.concatenate([delta_top_func(x_top[::-1]),delta_bot_func(x_bot)])
    
    tr_crit_top_func      = interp1d( x_t_top[::-1], tr_crit_top[::-1]) 
    tr_crit_bot_func      = interp1d( x_t_bot, tr_crit_bot) 
    Re_theta_t_top_func   = interp1d(x_t_top[::-1],  Re_theta_t_top[::-1])
    Re_theta_t_bot_func   = interp1d(x_t_bot,Re_theta_t_bot)
    Re_theta_t_vals       = np.concatenate([Re_theta_t_top_func(x_top[::-1]), Re_theta_t_bot_func(x_bot)])
    tr_crit_vals          = np.concatenate([tr_crit_top_func(x_top[::-1]),tr_crit_bot_func(x_bot)])
     
    s_vals                = np.zeros_like(x_vals)
    H_star_vals           = np.zeros_like(x_vals)
    P_vals                = np.zeros_like(x_vals)     
    m_vals                = np.zeros_like(x_vals)     
    K_vals                = np.zeros_like(x_vals)     
    tau_vals              = np.zeros_like(x_vals)  
    Di_val                = np.zeros_like(x_vals)   
     
    Cl,Cd,Cm = aero_coeff(x_vals,y_vals,Cp_vals,alpha,npanel)  
    
    airfoil_properties = Data(
        Cl         = Cl,
        Cd         = Cd,
        Cm         = Cm,
        normals    = normals,
        x          = x_vals,
        y          = y_vals,
        Cp         = Cp_vals,         
        Ue_Vinf    = Ve_vals,         
        dVe        = dVe_vals,         
        theta      = theta_vals,      
        delta_star = delta_star_vals, 
        delta      = delta_vals,
        H          = H_vals,          
        s          = s_vals,
        H_star     = H_star_vals, 
        P          = P_vals,      
        m          = m_vals,     
        K          = K_vals,     
        tau        = tau_vals,  
        Di         = Di_val,        
        Cf         = Cf_vals,          
        Re_theta_t = Re_theta_t_vals,
        tr_crit    = tr_crit_vals,     
        )  
    
    return  airfoil_properties
