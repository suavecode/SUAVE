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
def airfoil_analysis(airfoil_geometry,alpha,Re_L,npanel = 100,n_computation = 200, batch_analyis = True ):
    """This computed the aerodynamic polars as well as the boundary lawer properties of 
    an airfoil at a defined set of reynolds numbers and angle of attacks

    Assumptions:
    Mitchel Criteria used for transition

    Source:
    N/A

    Inputs: 
    airfoil_geometry  - airfoil geometry points 
    alpha             - angle of attacks
    Re_L              - Reynolds numbers
    npanel            - number of panels
    n_computation     - number of refined points of surface for boundary layer computation 
    
    Outputs: 
    airfoil_properties.
        AoA            - angle of attack
        Re             - Reynolds number
        Cl             - lift coefficients
        Cd             - drag coefficients
        Cm             - moment coefficients
        normals        - surface normals of airfoil
        x              - x coordinate points on airfoil 
        y              - y coordinate points on airfoil 
        x_bl           - x coordinate points on airfoil adjusted to include boundary layer
        y_bl           - y coordinate points on airfoil adjusted to include boundary layer
        Cp             - pressure coefficient distribution
        Ue_Vinf        - ratio of boundary layer edge velocity to freestream
        dVe            - derivative of boundary layer velocity
        theta          - momentum thickness 
        delta_star     - displacement thickness
        delta          - boundary layer thickness 
        H              - shape factor
        Cf             - local skin friction coefficient 
        Re_theta_t     - Reynolds Number as a function of theta transition location 
        tr_crit        - critical transition criteria	
                        
    Properties Used:
    N/A
    """   
    
    x_coord = np.delete( airfoil_geometry.x_coordinates[0][::-1], int(npanel/2))  # these are the vertices of each panel, len = 1 + npanel
    y_coord = np.delete( airfoil_geometry.y_coordinates[0][::-1], int(npanel/2))   
    
    # Begin by solving for velocity distribution at airfoil surface ucosg  inviscid panel simulation
    ## these are the locations (faces) where things are computed , len = n panel
    # dimension of vt = npanel x nalpha x nRe
    x,y,vt,cos_t,normals = hess_smith(x_coord,y_coord,alpha,Re_L,npanel)    
    
    nalpha           = len(alpha)
    nRe              = len(Re_L) 
    
    if not batch_analyis:
        if nalpha != nRe:
            raise AssertionError('Dimension of angle of attacks and Reynolds numbers must be equal')   
        else:
            nRe = 1
    
    # create datastructures        
    airfoil_Cl       = np.zeros((nalpha,nRe))
    airfoil_Cd       = np.zeros_like(airfoil_Cl)
    airfoil_Cm       = np.zeros_like(airfoil_Cl)           
    Cp_vals          = np.zeros((npanel,nalpha, nRe))
    Xbl_vals         = np.zeros((npanel,nalpha, nRe))
    Ybl_vals         = np.zeros((npanel,nalpha, nRe))
    Ve_vals          = np.zeros_like(Cp_vals)
    dVe_vals         = np.zeros_like(Cp_vals)
    theta_vals       = np.zeros_like(Cp_vals)
    delta_star_vals  = np.zeros_like(Cp_vals)
    H_vals           = np.zeros_like(Cp_vals)
    Cf_vals          = np.zeros_like(Cp_vals)
    delta_vals       = np.zeros_like(Cp_vals)
    Re_theta_t_vals  = np.zeros_like(Cp_vals)
    tr_crit_vals     = np.zeros_like(Cp_vals)  
    
    # convert to 2-D arrays
    X =  np.repeat(np.repeat(np.atleast_2d(x).T,nalpha,axis = 1)[:,:,np.newaxis], nRe, axis = 2)
    Y =  np.repeat(np.repeat(np.atleast_2d(y).T,nalpha,axis = 1)[:,:,np.newaxis], nRe, axis = 2)
    
    # ---------------------------------------------------------------------
    # Bottom surface of airfoil 
    # ---------------------------------------------------------------------    
    VT         = np.ma.masked_greater(vt,0 )
    VT_mask    = np.ma.masked_greater(vt,0 ).mask
    X_BOT_VALS = np.ma.array(X, mask = VT_mask)[::-1]
    Y_BOT      = np.ma.array(Y, mask = VT_mask)[::-1]
    
    X_BOT      = np.zeros_like(X_BOT_VALS)
    X_BOT[1:]  = np.cumsum(np.sqrt((X_BOT_VALS[1:] - X_BOT_VALS[:-1])**2 + (Y_BOT[1:] - Y_BOT[:-1])**2),axis = 0)
    first_idx  = np.ma.count_masked(X_BOT,axis = 0)
    prev_index = first_idx-1
    panel      = list(prev_index.flatten())
    aoas       = list(np.repeat(np.arange(nalpha),nRe))
    res        = list(np.tile(np.arange(nRe),nalpha) )
    X_BOT.mask[panel,aoas,res] = False
    
    # flow velocity and pressure of on botton surface 
    VE_BOT     = -VT[::-1] 
    CP_BOT     = 1 - VE_BOT**2  
    
    # velocity gradients on bottom surface  
    DVE_BOT       = np.zeros_like(X_BOT)
    DVE_BOT_TEMP  = np.diff(VE_BOT,axis = 0)/np.diff(X_BOT_VALS,axis = 0)
    a               = X_BOT[1:-1] - X_BOT[:-2]
    b               = X_BOT[2:]   - X_BOT[:-2]
    DVE_BOT[1:-1]   = (b*DVE_BOT_TEMP[:-1] + a*DVE_BOT_TEMP[1:])/(a+b) 
    DVE_BOT[-1,:,:] = DVE_BOT_TEMP[-1,:,:]  
    
    L_BOT           = L_BOT[-1,:,:] # x - location of stagnation point 
    
    for ial in range(nalpha):
        for iRe in range(nRe): 
            # Find stagnation point    
            if batch_analyis:
                iRe_     = iRe
                i_stag   = np.where(vt[:,ial,iRe_] > 0)[0][0]  
                Re_L_val = Re_L[iRe][0]
            else:
                iRe_     = ial
                i_stag   = np.where(vt[:,ial,iRe_] > 0)[0][0]  
                Re_L_val = Re_L[ial][0]
                
            # ---------------------------------------------------------------------
            # Bottom surface of airfoil 
            # ---------------------------------------------------------------------
            # x and y coordinates 
            x_bot_vals    = x[:i_stag][::-1]  # flip arrays on bottom surface (measured from stagnation point on bottom surface)
            y_bot         = y[:i_stag][::-1]  
            x_bot         = np.zeros_like(x_bot_vals)  
            x_bot[1:]     = np.cumsum(np.sqrt((x_bot_vals[1:] - x_bot_vals[:-1])**2 + (y_bot[1:] - y_bot[:-1])**2)) 
              
            ## flow velocity and pressure of on botton surface 
            Ve_bot        = -vt[:i_stag,ial,iRe_][::-1] # -a[:,0,0][::-1]  negative because the velocity is measured anti-clockwise around surface 
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
            #x_t_bot, theta_t_bot, del_star_t_bot, H_t_bot, cf_t_bot, Re_theta_t_bot, Re_x_t_bot,delta_t_bot= thwaites_method(0.000001, L_bot , Re_L_val, x_bot, Ve_bot, dVe_bot,n=n_computation)
            
            ## transition location  
            #tr_crit_bot     = Re_theta_t_bot - 1.174*(1 + 224000/Re_x_t_bot)*Re_x_t_bot**0.46                
            #tr_loc_vals     = np.where(tr_crit_bot > 0)[0] 
            #if len(tr_loc_vals) == 0:# no transition  
                #i_tr_bot =  len(tr_crit_bot) - 1
            #else: # transition 
                #i_tr_bot = tr_loc_vals[0]     
            
            #x_tr_bot        = x_t_bot[i_tr_bot] 
            #del_star_tr_bot = del_star_t_bot[i_tr_bot] 
            #theta_tr_bot    = theta_t_bot[i_tr_bot]    
            #delta_tr_bot    = delta_t_bot[i_tr_bot] 
            
            #x_h_bot, theta_h_bot, del_star_h_bot, H_h_bot, cf_h_bot, delta_h_bot = heads_method(delta_tr_bot,theta_tr_bot, del_star_tr_bot, L_bot - x_tr_bot,Re_L_val, x_bot - x_tr_bot, Ve_bot, dVe_bot,x_tr_bot,n=n_computation )
  
            ## determine if flow transitions  
            #x_bs            = np.concatenate([x_t_bot[:i_tr_bot], (x_h_bot + x_tr_bot)[i_tr_bot:]] )
            #theta_bot       = np.concatenate([theta_t_bot[:i_tr_bot] ,theta_h_bot[i_tr_bot:]] )
            #del_star_bot    = np.concatenate([del_star_t_bot[:i_tr_bot] ,del_star_h_bot[i_tr_bot:]] )
            #H_bot           = np.concatenate([H_t_bot[:i_tr_bot] ,H_h_bot[i_tr_bot:]] )
            #cf_bot          = np.concatenate([cf_t_bot[:i_tr_bot], cf_h_bot[i_tr_bot:]] )
            #delta_bot       = np.concatenate([delta_t_bot[:i_tr_bot], delta_h_bot[i_tr_bot:]]) 
            
            ## ---------------------------------------------------------------------
            ## Top surface of airfoil 
            ## ---------------------------------------------------------------------
            
            ##re-parameterise based on length of boundary for the top surface of the airfoil
            #x_top_vals    = x[i_stag:] 
            #y_top         = y[i_stag:] 
            #x_top         = np.zeros_like(x_top_vals)
            #x_top[1:]     = np.cumsum(np.sqrt((x_top_vals[1:] - x_top_vals[:-1])**2 + (y_top[1:] - y_top[:-1])**2))
            
            ## flow velocity and pressure on top surface 
            #Ve_top        = vt[i_stag:,ial,iRe_]  
            #Cp_top        = 1 - Ve_top**2
            
            ## velocity gradients on top surface 
            #dVe_top       = np.zeros_like(x_top_vals)
            #dVe_top_temp  = np.diff(Ve_top)/np.diff(x_top)
            #dVe_top[0]    = dVe_top_temp[0]
            #a             = x_top[1:-2] - x_top[:-3]
            #b             = x_top[2:-1] - x_top[:-3]
            #dVe_top[1:-2] = (b*dVe_top_temp[:-2] + a*dVe_top_temp[1:-1])/(a+b) 
            #dVe_top[-1]   = dVe_top_temp[-1]   
            #L_top         = x_top[-1]  
            
            ## laminar boundary layer properties using thwaites method 
            #x_t_top, theta_t_top, del_star_t_top, H_t_top, cf_t_top, Re_theta_t_top, Re_x_t_top, delta_t_top = thwaites_method(0.000001,L_top,Re_L_val, x_top, Ve_top, dVe_top,n=n_computation) 
            
            ## Mitchel's transition criteria (can often give nonsensical results) 
            #tr_crit_top     = Re_theta_t_top - 1.174*(1 + 224000/Re_x_t_top)*Re_x_t_top**0.46     
            #tr_loc_vals     = np.where(tr_crit_top > 0)[0] 
            #if len(tr_loc_vals) == 0:  # manual trip point at stagnation point (fully turbulent)
                #i_tr_top = 0
            #else: # transition
                #i_tr_top = tr_loc_vals[0]    
                
            #x_tr_top        = x_t_top[i_tr_top] 
            #del_star_tr_top = del_star_t_top[i_tr_top] 
            #theta_tr_top    = theta_t_top[i_tr_top]    
            #delta_tr_top    = delta_t_top[i_tr_top] 
            
            #x_h_top, theta_h_top, del_star_h_top, H_h_top, cf_h_top, delta_h_top = heads_method(delta_tr_top,theta_tr_top, del_star_tr_top, L_top - x_tr_top, Re_L_val, x_top - x_tr_top, Ve_top, dVe_top,x_tr_top,n=n_computation)
            
            ## determine if flow transitions  
            #x_ts            = np.concatenate([x_t_top[:i_tr_top], (x_h_top + x_tr_top)[i_tr_top:]])
            #theta_top       = np.concatenate([theta_t_top[:i_tr_top] ,theta_h_top[i_tr_top:]])
            #del_star_top    = np.concatenate([del_star_t_top[:i_tr_top], del_star_h_top[i_tr_top:]])
            #H_top           = np.concatenate([H_t_top[:i_tr_top] ,H_h_top[i_tr_top:]])
            #cf_top          = np.concatenate([cf_t_top[:i_tr_top], cf_h_top[i_tr_top:]])  
            #delta_top       = np.concatenate([delta_t_top[:i_tr_top], delta_h_top[i_tr_top:]]) 
            
            ## ---------------------------------------------------------------------
            ## Concatenate Upper and Lower Surface Data 
            ## ---------------------------------------------------------------------   
            
            #x_vals                = x[::-1]  
            #y_vals                = y[::-1]  
            
            #theta_top_func        = interp1d(x_ts[::-1], theta_top[::-1], fill_value='extrapolate')
            #theta_bot_func        = interp1d(x_bs,theta_bot, fill_value='extrapolate') 
            #delta_star_top_func   = interp1d(x_ts[::-1] ,del_star_top[::-1], fill_value='extrapolate')
            #delta_star_bot_func   = interp1d(x_bs,del_star_bot, fill_value='extrapolate') 
            #H_top_func            = interp1d(x_ts[::-1],H_top[::-1], fill_value='extrapolate')
            #H_bot_func            = interp1d(x_bs,H_bot, fill_value='extrapolate')   
            #Cf_top_func           = interp1d(x_ts[::-1],cf_top[::-1], fill_value='extrapolate')
            #Cf_bot_func           = interp1d(x_bs,cf_bot, fill_value='extrapolate') 
            #delta_top_func        = interp1d(x_ts[::-1],delta_top[::-1], fill_value='extrapolate')
            #delta_bot_func        = interp1d(x_bs,delta_bot, fill_value='extrapolate')     
            #tr_crit_top_func      = interp1d( x_t_top[::-1], tr_crit_top[::-1], fill_value='extrapolate') 
            #tr_crit_bot_func      = interp1d(x_t_bot, tr_crit_bot, fill_value='extrapolate') 
            #Re_theta_t_top_func   = interp1d(x_t_top[::-1],  Re_theta_t_top[::-1], fill_value='extrapolate')
            #Re_theta_t_bot_func   = interp1d(x_t_bot,Re_theta_t_bot, fill_value='extrapolate')
                                  

            #Cp_vals[:,ial,iRe]         = np.concatenate([Cp_top[::-1],Cp_bot])
            #Ve_vals[:,ial,iRe]         = np.concatenate([Ve_top[::-1],Ve_bot])
            #dVe_vals[:,ial,iRe]        = np.concatenate([dVe_top[::-1],dVe_bot ]) 
            #theta_vals[:,ial,iRe]      = np.concatenate([theta_top_func(x_top[::-1]),theta_bot_func(x_bot)])
            #delta_star_vals[:,ial,iRe] = np.concatenate([delta_star_top_func(x_top[::-1]),delta_star_bot_func(x_bot)])
            #H_vals[:,ial,iRe]          = np.concatenate([H_top_func(x_top[::-1]),H_bot_func(x_bot)])
            #Cf_vals[:,ial,iRe]         = np.concatenate([Cf_top_func(x_top[::-1]),Cf_bot_func(x_bot)])
            #delta_vals[:,ial,iRe]      = np.concatenate([delta_top_func(x_top[::-1]),delta_bot_func(x_bot)]) 
            #Re_theta_t_vals[:,ial,iRe] = np.concatenate([Re_theta_t_top_func(x_top[::-1]), Re_theta_t_bot_func(x_bot)])
            #tr_crit_vals[:,ial,iRe]    = np.concatenate([tr_crit_top_func(x_top[::-1]),tr_crit_bot_func(x_bot)]) 
             
            ## -------------------------------------------------------------------------------------------------
            ## Compute effective surface of airfoil with boundary layer and recompute aerodynamic properties 
            ## -------------------------------------------------------------------------------------------------
            #delta_vals        = np.nan_to_num(delta_vals)
            #new_y_coord       = np.zeros(npanel) 
            #flag              = (y_vals < 0)
            #new_y_coord       = y_vals        + delta_vals[:,ial,iRe]*(normals[:,1]) 
            #new_y_coord[flag] = y_vals[flag]  + delta_vals[:,ial,iRe][flag]*(normals[flag][:,1])  
            #new_x_coord       = x_vals        + delta_vals[:,ial,iRe]*(normals[:,0]) 
            #new_x_coord[flag] = x_vals[flag]  + delta_vals[:,ial,iRe][flag]*(normals[flag][:,0])  
            #y_coord           = np.insert(new_y_coord, int(npanel/2) ,0)[::-1]
            #x_coord           = np.insert(new_x_coord, int(npanel/2) ,0)[::-1]
            #x_coord           = x_coord - min(x_coord) 
            
            #new_x,new_y,new_vt,new_cos_t,new_normals = hess_smith(x_coord,y_coord,np.atleast_2d(alpha[ial]),np.atleast_2d(Re_L[iRe]),npanel)                
            #x_bl_vals             = new_x[::-1]  
            #y_bl_vals             = new_y[::-1]  
            #Xbl_vals[:,ial,iRe]   = x_bl_vals
            #Ybl_vals[:,ial,iRe]   = y_bl_vals
            
            ## Find stagnation point
            #i_stag             = np.where(new_vt[:,0,0] > 0)[0][0]  
                
            ## Bottom surface of airfoil  
            #new_x_bot_vals    = new_x[:i_stag][::-1]  # flip arrays on bottom surface (measured from stagnation point on bottom surface)
            #new_y_bot         = new_y[:i_stag][::-1]  
            #new_x_bot         = np.zeros_like(new_x_bot_vals )  
            #new_x_bot[1:]     = np.cumsum(np.sqrt((new_x_bot_vals[1:] - new_x_bot_vals [:-1])**2 + (new_y_bot[1:] - new_y_bot[:-1])**2)) 
              
            ## flow velocity and pressure of on botton surface 
            #new_Ve_bot        = -new_vt[:i_stag,0,0][::-1] # negative because the velocity is measured anti-clockwise around surface 
            #new_Cp_bot        = 1 - new_Ve_bot**2 
                 
            ## Top surface of airfoil  
            ##re-parameterise based on length of boundary for the top surface of the airfoil
            #new_x_top_vals    = new_x[i_stag:] 
            #new_y_top         = new_y[i_stag:] 
            #new_x_top         = np.zeros_like(new_x_top_vals)
            #new_x_top[1:]     = np.cumsum(np.sqrt((new_x_top_vals[1:] - new_x_top_vals[:-1])**2 + (new_y_top[1:] - new_y_top[:-1])**2))
            
            ## flow velocity and pressure on top surface 
            #new_Ve_top        = new_vt[i_stag:,0,0]  
            #new_Cp_top        = 1 - new_Ve_top**2
            
            #new_Cp_vals       = np.concatenate([new_Cp_top[::-1],new_Cp_bot])
            
            #Cl,Cd,Cm = aero_coeff(x_vals,y_vals,new_Cp_vals ,alpha[ial],npanel) 
            
            #airfoil_Cl[ial,iRe] = Cl
            #airfoil_Cd[ial,iRe] = Cd
            #airfoil_Cm[ial,iRe] = Cm 
    
    airfoil_properties = Data(
        AoA        = alpha,
        Re         = Re_L,
        Cl         = airfoil_Cl,
        Cd         = airfoil_Cd,
        Cm         = airfoil_Cm,
        normals    = normals,
        x          = x_vals,
        y          = y_vals,
        x_bl       = Xbl_vals,
        y_bl       = Ybl_vals,
        Cp         = Cp_vals,         
        Ue_Vinf    = Ve_vals,         
        dVe        = dVe_vals,         
        theta      = theta_vals,      
        delta_star = delta_star_vals, 
        delta      = delta_vals,
        H          = H_vals,               
        Cf         = Cf_vals,          
        Re_theta_t = Re_theta_t_vals,
        tr_crit    = tr_crit_vals,     
        )  
    
    return  airfoil_properties 