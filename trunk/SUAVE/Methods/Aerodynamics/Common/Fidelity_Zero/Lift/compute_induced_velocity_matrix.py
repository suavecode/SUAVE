## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# compute_induced_velocity_matrix.py
# 
# Created:  May 2018, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import SUAVE
import numpy as np
from SUAVE.Core import Units , Data

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def compute_induced_velocity_matrix(data,n_sw,n_cw,theta_w):     
    # unpack 
    XAH  = data.XAH 
    YAH  = data.YAH 
    ZAH  = data.ZAH 
    XBH  = data.XBH 
    YBH  = data.YBH 
    ZBH  = data.ZBH 
    XCH  = data.XCH 
    YCH  = data.YCH 
    ZCH  = data.ZCH 
          
    XA1  = data.XA1 
    YA1  = data.YA1 
    ZA1  = data.ZA1 
    XA2  = data.XA2 
    YA2  = data.YA2 
    ZA2  = data.ZA2 
          
    XB1  = data.XB1 
    YB1  = data.YB1 
    ZB1  = data.ZB1 
    XB2  = data.XB2 
    YB2  = data.YB2 
    ZB2  = data.ZB2 
          
    XAC  = data.XAC 
    YAC  = data.YAC 
    ZAC  = data.ZAC 
    XBC  = data.XBC 
    YBC  = data.YBC 
    ZBC  = data.ZBC 
    XC   = data.XC  
    YC   = data.YC  
    ZC   = data.ZC   
    n_w  = data.n_w
    
    # ---------------------------------------------------------------------------------------
    # Compute velocity induced by horseshoe vortex segments on every control point by every panel
    # --------------------------------------------------------------------------------------- 
    n_cp       = n_w*n_cw*n_sw # total number of control points 
    C_AB_34_ll = np.zeros((n_cp,n_cp,3))
    C_AB_ll    = np.zeros((n_cp,n_cp,3))
    C_AB_34_rl = np.zeros((n_cp,n_cp,3))    
    C_AB_rl    = np.zeros((n_cp,n_cp,3))
    C_AB_bv    = np.zeros((n_cp,n_cp,3))
    C_Ainf     = np.zeros((n_cp,n_cp,3))
    C_Binf     = np.zeros((n_cp,n_cp,3))
    C_mn = np.zeros((n_cp,n_cp,3))
    
    for m in range(n_cp): # control point m
        sw_idx = 1
        cw_idx = 0
        for n in range(n_cp): # horseshe vortex n  	         
            # trailing vortices  
            XC_hat  = np.cos(theta_w)*XC[m] + np.sin(theta_w)*ZC[m]
            YC_hat  = YC[m]
            ZC_hat  = -np.sin(theta_w)*XC[m] + np.cos(theta_w)*ZC[m] 

            XA2_hat =  np.cos(theta_w)*XA2[(sw_idx*n_cw)-1]  + np.sin(theta_w)*ZA2[(sw_idx*n_cw)-1]  
            YA2_hat =  YA2[(sw_idx*n_cw)-1]            
            ZA2_hat = -np.sin(theta_w)*XA2[(sw_idx*n_cw)-1]  + np.cos(theta_w)*ZA2[(sw_idx*n_cw)-1]            

            XB2_hat =  np.cos(theta_w)*XB2[(sw_idx*n_cw)-1]  + np.sin(theta_w)*ZB2[(sw_idx*n_cw)-1]  
            YB2_hat =  YB2[(sw_idx*n_cw)-1]            
            ZB2_hat = -np.sin(theta_w)*XB2[(sw_idx*n_cw)-1]  + np.cos(theta_w)*ZB2[(sw_idx*n_cw)-1]         

            # starboard (right) wing 
            if YBH[n] > 0:
                # compute influence of bound vortices 
                C_AB_bv[m,n] = vortex(XC[m], YC[m], ZC[m], XAH[n], YAH[n], ZAH[n], XBH[n], YBH[n], ZBH[n])   

                # compute influence of 3/4 left legs
                C_AB_34_ll[m,n] = vortex(XC[m], YC[m], ZC[m], XA2[n], YA2[n], ZA2[n], XAH[n], YAH[n], ZAH[n])      

                # compute influence of whole panel left legs 
                C_AB_ll[m,n]   =  vortex(XC[m], YC[m], ZC[m], XA2[n], YA2[n], ZA2[n], XA1[n], YA1[n], ZA1[n])      

                # compute influence of 3/4 right legs
                C_AB_34_rl[m,n] = vortex(XC[m], YC[m], ZC[m], XBH[n], YBH[n], ZBH[n], XB2[n], YB2[n], ZB2[n])      

                # compute influence of whole right legs 
                C_AB_rl[m,n] = vortex(XC[m], YC[m], ZC[m], XB1[n], YB1[n], ZB1[n], XB2[n], YB2[n], ZB2[n])    

                # velocity induced by left leg of vortex (A to inf)
                C_Ainf[m,n]  = vortex_to_inf_l(XC_hat, YC_hat, ZC_hat, XA2_hat, YA2_hat, ZA2_hat,theta_w)     

                # velocity induced by right leg of vortex (B to inf)
                C_Binf[m,n]  = vortex_to_inf_r(XC_hat, YC_hat, ZC_hat, XB2_hat, YB2_hat, ZB2_hat,theta_w) 

            # port (left) wing 
            else: 
                # compute influence of bound vortices 
                C_AB_bv[m,n]   = vortex(XC[m], YC[m], ZC[m], XBH[n], YBH[n], ZBH[n], XAH[n], YAH[n], ZAH[n])                       

                # compute influence of 3/4 left legs
                C_AB_34_ll[m,n] = vortex(XC[m], YC[m], ZC[m], XB2[n], YB2[n], ZB2[n], XBH[n], YBH[n], ZBH[n])        

                # compute influence of whole panel left legs 
                C_AB_ll[m,n]    =  vortex(XC[m], YC[m], ZC[m], XB2[n], YB2[n], ZB2[n], XB1[n], YB1[n], ZB1[n])      

                # compute influence of 3/4 right legs
                C_AB_34_rl[m,n] = vortex(XC[m], YC[m], ZC[m], XAH[n], YAH[n], ZAH[n], XA2[n], YA2[n], ZA2[n]) 

                # compute influence of whole right legs 
                C_AB_rl[m,n] =  vortex(XC[m], YC[m], ZC[m], XA1[n], YA1[n], ZA1[n], XA2[n], YA2[n], ZA2[n])   

                # velocity induced by left leg of vortex (A to inf)
                C_Ainf[m,n]  = vortex_to_inf_l(XC_hat, YC_hat, ZC_hat, XB2_hat, YB2_hat, ZB2_hat,theta_w)  

                # velocity induced by right leg of vortex (B to inf)
                C_Binf[m,n]  =  vortex_to_inf_r(XC_hat, YC_hat, ZC_hat, XA2_hat, YA2_hat, ZA2_hat,theta_w)   
            cw_idx += 1
            if cw_idx == n_cw:
                cw_idx = 0
                sw_idx += 1
                
    # Summation of Influences
    # Add in the regular effects except AB_rl_ll   
                
    # Add the right and left influences seperateky
    C_AB_llrl_roll = C_AB_ll+C_AB_rl
    
    # Prime the arrays
    mask = np.ones_like(C_mn)
    C_AB_llrl = np.zeros_like(C_mn)

    for idx in range(n_cw-1):    
        
        # Make a mask
        mask[:,n_cw-idx-1::n_cw,:]= np.zeros_like(mask[:,n_cw-idx-1::n_cw,:])  
        # Roll it over to the next component
        
        C_AB_llrl_roll = np.roll(C_AB_llrl_roll,-1,axis=1)   
        
        # Add in the components that we need
        C_AB_llrl = C_AB_llrl+ C_AB_llrl_roll*mask
   

    # Add all the sources together
    C_mn = C_AB_34_ll + C_AB_bv + C_AB_34_rl + C_Ainf + C_Binf + C_AB_llrl
    
    return C_mn

# -------------------------------------------------------------------------------
# vortex strength computation
# -------------------------------------------------------------------------------
def vortex(X,Y,Z,X1,Y1,Z1,X2,Y2,Z2):
    R1R2X  =   (Y-Y1)*(Z-Z2) - (Z-Z1)*(Y-Y2) 
    R1R2Y  = -((X-X1)*(Z-Z2) - (Z-Z1)*(X-X2))
    R1R2Z  =   (X-X1)*(Y-Y2) - (Y-Y1)*(X-X2)
    SQUARE = R1R2X**2 + R1R2Y**2 + R1R2Z**2
    R1     = np.sqrt((X-X1)**2 + (Y-Y1)**2 + (Z-Z1)**2)
    R2     = np.sqrt((X-X2)**2 + (Y-Y2)**2 + (Z-Z2)**2)
    R0R1   = (X2-X1)*(X-X1) + (Y2-Y1)*(Y-Y1) + (Z2-Z1)*(Z-Z1)
    R0R2   = (X2-X1)*(X-X2) + (Y2-Y1)*(Y-Y2) + (Z2-Z1)*(Z-Z2)
    RVEC   = np.array([R1R2X,R1R2Y,R1R2Z])
    COEF   = (1/(4*np.pi))*(RVEC/SQUARE) * (R0R1/R1 - R0R2/R2)
    return COEF

def vortex_to_inf_l(X,Y,Z,X1,Y1,Z1,tw): 
    DENUM =  (Z-Z1)**2 + (Y1-Y)**2
    XVEC  = -(Y1-Y)*np.sin(tw)/DENUM
    YVEC  =  (Z-Z1)/DENUM
    ZVEC  =  (Y1-Y)*np.cos(tw)/DENUM
    BRAC  =  1 + ((X-X1) / (np.sqrt((X-X1)**2+ (Y-Y1)**2+ (Z-Z1)**2)))
    RVEC   = np.array([XVEC, YVEC, ZVEC])
    COEF  = (1/(4*np.pi))*RVEC*BRAC         
    return COEF

def vortex_to_inf_r(X,Y,Z,X1,Y1,Z1,tw):
    DENUM =  (Z-Z1)**2 + (Y1-Y)**2
    XVEC  = (Y1-Y)*np.sin(tw)/DENUM
    YVEC  = -(Z-Z1)/DENUM
    ZVEC  = -(Y1-Y)*np.cos(tw)/DENUM
    BRAC  =  1 + ((X-X1) / (np.sqrt((X-X1)**2+ (Y-Y1)**2+ (Z-Z1)**2)))
    RVEC   = np.array([XVEC, YVEC, ZVEC])
    COEF  = (1/(4*np.pi))*RVEC*BRAC         
    return COEF

