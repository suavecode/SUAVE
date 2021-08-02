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
def airfoil_analysis(airfoil_geometry,alpha,Re_L,npanel = 100,n_computation = 200, batch_analyis = True, airfoil_stations = None):
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
    batch_analyis     - boolean : If True: the specified number of angle of attacks and Reynolds
                                  numbers are used to create a table of 2-D results for each combination
                                  Note: Can only accomodate one airfoil
                                  
                                  If False:The airfoils specified are run and corresponding angle of attacks 
                                  and Reynolds numbers
                                  Note: The number of airfoils, angle of attacks and reynolds numbers must 
                                  all the same dimension 
    
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
    
    
    nalpha           = len(alpha)
    nRe              = len(Re_L) 
    
    if batch_analyis:
        x_coord    = np.delete( airfoil_geometry.x_coordinates[0][::-1], int(npanel/2))  # these are the vertices of each panel, len = 1 + npanel
        y_coord    = np.delete( airfoil_geometry.y_coordinates[0][::-1], int(npanel/2))    
    else:
        try: 
            nairfoil         = len(airfoil_stations) 
        except:
            raise AssertionError('Specifiy airfoil stations') 
        
        if (nalpha != nRe) and ( nairfoil!= nalpha):
            raise AssertionError('Dimension of angle of attacks,Reynolds numbers and airfoil stations must all be equal')    
        
        x_coord    = np.take(airfoil_geometry.x_coordinates,airfoil_stations,axis=0)  
        y_coord    = np.take(airfoil_geometry.y_coordinates,airfoil_stations,axis=0) 
        x_coord    = np.delete( x_coord[::-1], int(npanel/2))  
        y_coord    = np.delete( y_coord[::-1], int(npanel/2))     
   

    x_coord_3d = np.repeat(np.repeat(np.atleast_2d(x_coord).T,nalpha,axis = 1)[:,:,np.newaxis],nRe, axis = 2)
    y_coord_3d = np.repeat(np.repeat(np.atleast_2d(y_coord).T,nalpha,axis = 1)[:,:,np.newaxis],nRe, axis = 2)
        
    # Begin by solving for velocity distribution at airfoil surface ucosg  inviscid panel simulation
    ## these are the locations (faces) where things are computed , len = n panel
    # dimension of vt = npanel x nalpha x nRe
    X,Y,vt,cos_t,normals = hess_smith(x_coord_3d,y_coord_3d,alpha,Re_L,npanel,batch_analyis)  
    
    # Reynolds number 
    RE_L_VALS = np.repeat(Re_L.T,nalpha, axis = 0)
    
    # ---------------------------------------------------------------------
    # Bottom surface of airfoil 
    # ---------------------------------------------------------------------     
    VT              = np.ma.masked_greater(vt,0 )
    VT_mask         = np.ma.masked_greater(vt,0 ).mask
    X_BOT_VALS      = np.ma.array(X, mask = VT_mask)[::-1]
    Y_BOT           = np.ma.array(Y, mask = VT_mask)[::-1]
         
    X_BOT           = np.zeros_like(X_BOT_VALS)
    X_BOT[1:]       = np.cumsum(np.sqrt((X_BOT_VALS[1:] - X_BOT_VALS[:-1])**2 + (Y_BOT[1:] - Y_BOT[:-1])**2),axis = 0)
    first_idx       = np.ma.count_masked(X_BOT,axis = 0)
    mask_count      = np.ma.count(X_BOT,axis = 0)
    prev_index      = first_idx-1
    first_panel     = list(prev_index.flatten())
    last_panel      = list((first_idx-1 + mask_count).flatten())
    last_paneldve   = list((first_idx-2 + mask_count).flatten())
    aoas            = list(np.repeat(np.arange(nalpha),nRe))
    res             = list(np.tile(np.arange(nRe),nalpha) )
    X_BOT.mask[first_panel,aoas,res] = False
    
    # flow velocity and pressure of on botton surface 
    VE_BOT          = -VT[::-1] 
    CP_BOT          = 1 - VE_BOT**2  
    
    # velocity gradients on bottom surface  
    DVE_BOT                        = np.ma.zeros(np.shape(X_BOT)) 
    DVE_BOT_TEMP                   = np.diff(VE_BOT,axis = 0)/np.diff(X_BOT,axis = 0)
    a                              = X_BOT[1:-1] - X_BOT[:-2]
    b                              = X_BOT[2:]   - X_BOT[:-2]
    DVE_BOT[1:-1]                  = ((b*DVE_BOT_TEMP[:-1] + a*DVE_BOT_TEMP[1:])/(a+b)).data
    DVE_BOT.mask                   = X_BOT.mask
    DVE_BOT[first_panel,aoas,res]  = DVE_BOT_TEMP[first_panel,aoas,res]
    DVE_BOT[last_panel,aoas,res]   = DVE_BOT_TEMP[last_paneldve,aoas,res] 
    
    # x - location of stagnation point 
    L_BOT                          = X_BOT[-1,:,:]    
        
    # laminar boundary layer properties using thwaites method 
    BOT_T_RESULTS  = thwaites_method(nalpha,nRe, L_BOT , RE_L_VALS, X_BOT, VE_BOT, DVE_BOT,batch_analyis,
                                     THETA_0=0.000001,n=n_computation) 
    X_T_BOT          = BOT_T_RESULTS.X_T      
    THETA_T_BOT      = BOT_T_RESULTS.THETA_T     
    DELTA_STAR_T_BOT = BOT_T_RESULTS.DELTA_STAR_T  
    H_T_BOT          = BOT_T_RESULTS.H_T         
    CF_T_BOT         = BOT_T_RESULTS.CF_T        
    RE_THETA_T_BOT   = BOT_T_RESULTS.RE_THETA_T    
    RE_X_T_BOT       = BOT_T_RESULTS.RE_X_T      
    DELTA_T_BOT      = BOT_T_RESULTS.DELTA_T      
     
    # transition location  
    TR_CRIT_BOT     = RE_THETA_T_BOT - 1.174*(1 + 224000/RE_X_T_BOT)*RE_X_T_BOT**0.46  
    CRITERION_BOT   = np.ma.masked_greater(TR_CRIT_BOT,0 ) 
    if batch_analyis:
        mask_count  = np.ma.count(CRITERION_BOT,axis = 0)  
        mask_count[mask_count == n_computation] = n_computation-1
    else:
        mask_count  =  np.zeros((nalpha,nalpha)).astype(int) 
    transition_panel = list(mask_count.flatten()) 
    aoas             = list(np.repeat(np.arange(nalpha),nRe))
    res              = list(np.tile(np.arange(nRe),nalpha))
        
    X_TR_BOT          = X_T_BOT[transition_panel,aoas,res].reshape(nalpha,nRe)
    DELTA_STAR_TR_BOT = DELTA_STAR_T_BOT[transition_panel,aoas,res].reshape(nalpha,nRe)
    THETA_TR_BOT      = THETA_T_BOT[transition_panel,aoas,res].reshape(nalpha,nRe)    
    DELTA_TR_BOT      = DELTA_T_BOT[transition_panel,aoas,res].reshape(nalpha,nRe) 
    
    TURBULENT_SURF  = L_BOT.data  - X_TR_BOT
    TURBULENT_COORD = X_BOT       - X_TR_BOT
    
    # turbulent boundary layer properties using heads method 
    BOT_H_RESULTS   = heads_method(nalpha,nRe,DELTA_TR_BOT ,THETA_TR_BOT , DELTA_STAR_TR_BOT,
                                   TURBULENT_SURF, RE_L_VALS,TURBULENT_COORD, VE_BOT, DVE_BOT ,X_TR_BOT,
                                   batch_analyis,n=n_computation )
    
    X_H_BOT          = BOT_H_RESULTS.X_H      
    THETA_H_BOT      = BOT_H_RESULTS.THETA_H   
    DELTA_STAR_H_BOT = BOT_H_RESULTS.DELTA_STAR_H
    H_H_BOT          = BOT_H_RESULTS.H_H       
    CF_H_BOT         = BOT_H_RESULTS.CF_H    
    RE_THETA_H_BOT   = BOT_H_RESULTS.RE_THETA_H 
    RE_X_H_BOT       = BOT_H_RESULTS.RE_X_H         
    DELTA_H_BOT      = BOT_H_RESULTS.DELTA_H   
    
    # Apply Masks to surface vectors  
    X_T_BOT          = np.ma.array(X_T_BOT , mask = CRITERION_BOT.mask)
    THETA_T_BOT      = np.ma.array(THETA_T_BOT , mask = CRITERION_BOT.mask)   
    DELTA_STAR_T_BOT = np.ma.array(DELTA_STAR_T_BOT , mask = CRITERION_BOT.mask)
    H_T_BOT          = np.ma.array(H_T_BOT , mask = CRITERION_BOT.mask)       
    CF_T_BOT         = np.ma.array(CF_T_BOT , mask = CRITERION_BOT.mask)      
    RE_THETA_T_BOT   = np.ma.array(RE_THETA_T_BOT , mask = CRITERION_BOT.mask) 
    RE_X_T_BOT       = np.ma.array(RE_X_T_BOT  , mask = CRITERION_BOT.mask)   
    DELTA_T_BOT      = np.ma.array(DELTA_T_BOT , mask = CRITERION_BOT.mask)    
                       
    X_H_BOT          = np.ma.array(X_H_BOT , mask = ~CRITERION_BOT.mask)       
    THETA_H_BOT      = np.ma.array(THETA_H_BOT , mask = ~CRITERION_BOT.mask)   
    DELTA_STAR_H_BOT = np.ma.array(DELTA_STAR_H_BOT , mask = ~CRITERION_BOT.mask)
    H_H_BOT          = np.ma.array(H_H_BOT  , mask = ~CRITERION_BOT.mask)      
    CF_H_BOT         = np.ma.array(CF_H_BOT , mask = ~CRITERION_BOT.mask)  
    RE_THETA_H_BOT   = np.ma.array(RE_THETA_H_BOT  , mask = ~CRITERION_BOT .mask)    
    RE_X_H_BOT       = np.ma.array(RE_X_H_BOT  , mask = ~CRITERION_BOT .mask)        
    DELTA_H_BOT      = np.ma.array(DELTA_H_BOT , mask = ~CRITERION_BOT.mask)   
        
    
    # Concatenate vectors
    X_H_BOT_MOD = X_H_BOT.data + X_TR_BOT.data
    X_H_BOT_MOD = np.ma.array(X_H_BOT_MOD, mask = X_H_BOT.mask)
    
    X_BOT_SURF           = np.ma.concatenate([X_T_BOT, X_H_BOT_MOD], axis = 0 ) 
    THETA_BOT_SURF       = np.ma.concatenate([THETA_T_BOT,THETA_H_BOT ], axis = 0)
    DELTA_STAR_BOT_SURF  = np.ma.concatenate([DELTA_STAR_T_BOT,DELTA_STAR_H_BOT], axis = 0)
    H_BOT_SURF           = np.ma.concatenate([H_T_BOT,H_H_BOT], axis = 0)
    CF_BOT_SURF          = np.ma.concatenate([CF_T_BOT,CF_H_BOT], axis = 0)
    RE_THETA_BOT_SURF    = np.ma.concatenate([RE_THETA_T_BOT,RE_THETA_H_BOT], axis = 0)
    RE_X_BOT_SURF        = np.ma.concatenate([RE_X_T_BOT,RE_X_H_BOT], axis = 0)    
    DELTA_BOT_SURF       = np.ma.concatenate([DELTA_T_BOT,DELTA_H_BOT], axis = 0) 
    
    X_BOT_SURF           = X_BOT_SURF[X_BOT_SURF.mask == False ].reshape(n_computation,nalpha,nRe).data
    THETA_BOT_SURF       = THETA_BOT_SURF[THETA_BOT_SURF.mask == False ].reshape(n_computation,nalpha,nRe).data
    DELTA_STAR_BOT_SURF  = DELTA_STAR_BOT_SURF[DELTA_STAR_BOT_SURF.mask == False ].reshape(n_computation,nalpha,nRe).data
    H_BOT_SURF           = H_BOT_SURF[H_BOT_SURF.mask == False ].reshape(n_computation,nalpha,nRe).data
    CF_BOT_SURF          = CF_BOT_SURF[CF_BOT_SURF.mask == False ].reshape(n_computation,nalpha,nRe).data
    RE_THETA_BOT_SURF    = RE_THETA_BOT_SURF[RE_THETA_BOT_SURF.mask == False ].reshape(n_computation,nalpha,nRe).data
    RE_X_BOT_SURF        = RE_X_BOT_SURF[RE_X_BOT_SURF.mask == False ].reshape(n_computation,nalpha,nRe).data    
    DELTA_BOT_SURF       = DELTA_BOT_SURF[DELTA_BOT_SURF.mask == False ].reshape(n_computation,nalpha,nRe).data
    
    
    # ------------------------------------------------------------------------------------------------------
    # Top surface of airfoil 
    # ------------------------------------------------------------------------------------------------------ 
    VT              = np.ma.masked_less(vt,0 )
    VT_mask         = np.ma.masked_less(vt,0 ).mask
    X_TOP_VALS      = np.ma.array(X, mask = VT_mask) 
    Y_TOP           = np.ma.array(Y, mask = VT_mask) 

    X_TOP           = np.zeros_like(X_TOP_VALS)
    X_TOP[1:]       = np.cumsum(np.sqrt((X_TOP_VALS[1:] - X_TOP_VALS[:-1])**2 + (Y_TOP[1:] - Y_TOP[:-1])**2),axis = 0)
    first_idx       = np.ma.count_masked(X_TOP,axis = 0)
    mask_count      = np.ma.count(X_TOP,axis = 0)
    prev_index      = first_idx-1
    first_panel     = list(prev_index.flatten())
    last_panel      = list((first_idx-1 + mask_count).flatten())
    last_paneldve   = list((first_idx-2 + mask_count).flatten())
    aoas            = list(np.repeat(np.arange(nalpha),nRe))
    res             = list(np.tile(np.arange(nRe),nalpha) )
    X_TOP.mask[first_panel,aoas,res] = False

    # flow velocity and pressure of on botton surface 
    VE_TOP          = VT 
    CP_TOP          = 1 - VE_TOP**2  
    
    # velocity gradients on bottom surface  
    DVE_TOP                        = np.ma.zeros(np.shape(X_TOP)) 
    DVE_TOP_TEMP                   = np.diff(VE_TOP,axis = 0)/np.diff(X_TOP,axis = 0)
    a                              = X_TOP[1:-1] - X_TOP[:-2]
    b                              = X_TOP[2:]   - X_TOP[:-2]
    DVE_TOP[1:-1]                  = ((b*DVE_TOP_TEMP[:-1] + a*DVE_TOP_TEMP[1:])/(a+b)).data
    DVE_TOP.mask                   = X_TOP.mask
    DVE_TOP[first_panel,aoas,res]  = DVE_TOP_TEMP[first_panel,aoas,res]
    DVE_TOP[last_panel,aoas,res]   = DVE_TOP_TEMP[last_paneldve,aoas,res] 

    # x - location of stagnation point 
    L_TOP                          = X_TOP[-1,:,:]    

    # laminar boundary layer properties using thwaites method 
    BOT_T_RESULTS    = thwaites_method(nalpha,nRe, L_TOP , RE_L_VALS, X_TOP, VE_TOP, DVE_TOP,batch_analyis,
                                     THETA_0=0.000001,n=n_computation) 
    X_T_TOP          = BOT_T_RESULTS.X_T      
    THETA_T_TOP      = BOT_T_RESULTS.THETA_T     
    DELTA_STAR_T_TOP = BOT_T_RESULTS.DELTA_STAR_T  
    H_T_TOP          = BOT_T_RESULTS.H_T         
    CF_T_TOP         = BOT_T_RESULTS.CF_T        
    RE_THETA_T_TOP   = BOT_T_RESULTS.RE_THETA_T    
    RE_X_T_TOP       = BOT_T_RESULTS.RE_X_T      
    DELTA_T_TOP      = BOT_T_RESULTS.DELTA_T      

    # transition location  
    TR_CRIT_TOP       = RE_THETA_T_TOP - 1.174*(1 + 224000/RE_X_T_TOP)*RE_X_T_TOP**0.46  
    CRITERION_TOP     = np.ma.masked_greater(TR_CRIT_TOP,0 )
  
    mask_count        = np.ma.count_masked(CRITERION_TOP,axis = 0) 
     
    transition_panel  = list(mask_count.flatten()) 
    aoas              = list(np.repeat(np.arange(nalpha),nRe))
    res               = list(np.tile(np.arange(nRe),nalpha) )
  
    X_TR_TOP          = X_T_TOP[transition_panel,aoas,res].reshape(nalpha,nRe)
    DELTA_STAR_TR_TOP = DELTA_STAR_T_TOP[transition_panel,aoas,res].reshape(nalpha,nRe)
    THETA_TR_TOP      = THETA_T_TOP[transition_panel,aoas,res].reshape(nalpha,nRe)    
    DELTA_TR_TOP      = DELTA_T_TOP[transition_panel,aoas,res].reshape(nalpha,nRe) 
  
    TURBULENT_SURF    = L_TOP.data  - X_TR_TOP
    TURBULENT_COORD   = X_TOP  - X_TR_TOP

    # turbulent boundary layer properties using heads method 
    TOP_H_RESULTS     = heads_method(nalpha,nRe,DELTA_TR_TOP ,THETA_TR_TOP , DELTA_STAR_TR_TOP,
                                   TURBULENT_SURF, RE_L_VALS,TURBULENT_COORD, VE_TOP, DVE_TOP ,X_TR_TOP,
                                   batch_analyis,n=n_computation )

    X_H_TOP          = TOP_H_RESULTS.X_H      
    THETA_H_TOP      = TOP_H_RESULTS.THETA_H   
    DELTA_STAR_H_TOP = TOP_H_RESULTS.DELTA_STAR_H
    H_H_TOP          = TOP_H_RESULTS.H_H       
    CF_H_TOP         = TOP_H_RESULTS.CF_H        
    RE_THETA_H_TOP   = TOP_H_RESULTS.RE_THETA_H  
    RE_X_H_TOP       = TOP_H_RESULTS.RE_X_H     
    DELTA_H_TOP      = TOP_H_RESULTS.DELTA_H 
    
    # Apply Masks to surface vectors  
    X_T_TOP          = np.ma.array(X_T_TOP , mask = CRITERION_TOP.mask)
    THETA_T_TOP      = np.ma.array(THETA_T_TOP , mask = CRITERION_TOP.mask)   
    DELTA_STAR_T_TOP = np.ma.array(DELTA_STAR_T_TOP , mask = CRITERION_TOP.mask)
    H_T_TOP          = np.ma.array(H_T_TOP , mask = CRITERION_TOP.mask)       
    CF_T_TOP         = np.ma.array(CF_T_TOP , mask = CRITERION_TOP.mask)      
    RE_THETA_T_TOP   = np.ma.array(RE_THETA_T_TOP , mask = CRITERION_TOP.mask) 
    RE_X_T_TOP       = np.ma.array(RE_X_T_TOP  , mask = CRITERION_TOP.mask)   
    DELTA_T_TOP      = np.ma.array(DELTA_T_TOP , mask = CRITERION_TOP.mask)    
                       
    X_H_TOP          = np.ma.array(X_H_TOP , mask = ~CRITERION_TOP.mask)       
    THETA_H_TOP      = np.ma.array(THETA_H_TOP , mask = ~CRITERION_TOP.mask)   
    DELTA_STAR_H_TOP = np.ma.array(DELTA_STAR_H_TOP , mask = ~CRITERION_TOP.mask)
    H_H_TOP          = np.ma.array(H_H_TOP  , mask = ~CRITERION_TOP.mask)      
    CF_H_TOP         = np.ma.array(CF_H_TOP , mask = ~CRITERION_TOP.mask)    
    RE_THETA_H_TOP   = np.ma.array(RE_THETA_H_TOP , mask = ~CRITERION_TOP.mask)    
    RE_X_H_TOP       = np.ma.array(RE_X_H_TOP , mask = ~CRITERION_TOP.mask)    
    DELTA_H_TOP      = np.ma.array(DELTA_H_TOP , mask = ~CRITERION_TOP.mask)   
    
    # Concatenate laminart and turbulent vectors
    X_H_TOP_MOD          = X_H_TOP.data + X_TR_TOP.data
    X_H_TOP_MOD          = np.ma.array(X_H_TOP_MOD, mask = X_H_TOP.mask)
    
    X_TOP_SURF           = np.ma.concatenate([X_T_TOP, X_H_TOP_MOD], axis = 0 ) 
    THETA_TOP_SURF       = np.ma.concatenate([THETA_T_TOP,THETA_H_TOP ], axis = 0)
    DELTA_STAR_TOP_SURF  = np.ma.concatenate([DELTA_STAR_T_TOP,DELTA_STAR_H_TOP], axis = 0)
    H_TOP_SURF           = np.ma.concatenate([H_T_TOP,H_H_TOP], axis = 0)
    CF_TOP_SURF          = np.ma.concatenate([CF_T_TOP,CF_H_TOP], axis = 0)
    RE_THETA_TOP_SURF    = np.ma.concatenate([RE_THETA_T_TOP,RE_THETA_H_TOP], axis = 0)
    RE_X_TOP_SURF        = np.ma.concatenate([RE_X_T_TOP,RE_X_H_TOP], axis = 0)
    DELTA_TOP_SURF       = np.ma.concatenate([DELTA_T_TOP,DELTA_H_TOP], axis = 0) 
    
    # Use mask to trim vector down to from 2*n_computation  to n_computation size
    X_TOP_SURF           = X_TOP_SURF[X_TOP_SURF.mask == False ].reshape(n_computation,nalpha,nRe).data
    THETA_TOP_SURF       = THETA_TOP_SURF[THETA_TOP_SURF.mask == False ].reshape(n_computation,nalpha,nRe).data
    DELTA_STAR_TOP_SURF  = DELTA_STAR_TOP_SURF[DELTA_STAR_TOP_SURF.mask == False ].reshape(n_computation,nalpha,nRe).data
    H_TOP_SURF           = H_TOP_SURF[H_TOP_SURF.mask == False ].reshape(n_computation,nalpha,nRe).data
    CF_TOP_SURF          = CF_TOP_SURF[CF_TOP_SURF.mask == False ].reshape(n_computation,nalpha,nRe).data
    RE_THETA_TOP_SURF    = RE_THETA_TOP_SURF[RE_THETA_TOP_SURF.mask == False ].reshape(n_computation,nalpha,nRe).data
    RE_X_TOP_SURF        = RE_X_TOP_SURF[RE_X_TOP_SURF.mask == False ].reshape(n_computation,nalpha,nRe).data
    DELTA_TOP_SURF       = DELTA_TOP_SURF[DELTA_TOP_SURF.mask == False ].reshape(n_computation,nalpha,nRe).data    
    
    # ------------------------------------------------------------------------------------------------------
    # Interpolate to trim vector down to n_computation size  
    # ------------------------------------------------------------------------------------------------------      
    THETA      = interpolate_values(X_BOT,X_TOP,X_BOT_SURF,X_TOP_SURF,THETA_BOT_SURF,THETA_TOP_SURF,npanel,nalpha,nRe)
    DELTA_STAR = interpolate_values(X_BOT,X_TOP,X_BOT_SURF,X_TOP_SURF,DELTA_STAR_BOT_SURF,DELTA_STAR_TOP_SURF,npanel,nalpha,nRe) 
    H          = interpolate_values(X_BOT,X_TOP,X_BOT_SURF,X_TOP_SURF,H_BOT_SURF,H_TOP_SURF,npanel,nalpha,nRe)  
    CF         = interpolate_values(X_BOT,X_TOP,X_BOT_SURF,X_TOP_SURF,CF_BOT_SURF,CF_TOP_SURF,npanel,nalpha,nRe) 
    RE_THETA   = interpolate_values(X_BOT,X_TOP,X_BOT_SURF,X_TOP_SURF,RE_THETA_BOT_SURF,RE_THETA_TOP_SURF,npanel,nalpha,nRe)  
    RE_X       = interpolate_values(X_BOT,X_TOP,X_BOT_SURF,X_TOP_SURF,RE_X_BOT_SURF,RE_X_TOP_SURF,npanel,nalpha,nRe) 
    DELTA      = interpolate_values(X_BOT,X_TOP,X_BOT_SURF,X_TOP_SURF,DELTA_BOT_SURF,DELTA_TOP_SURF,npanel,nalpha,nRe)   
     
    VE_VALS    = np.ma.concatenate([VE_TOP,VE_BOT ], axis = 0)
    DVE_VALS   = np.ma.concatenate([DVE_TOP,DVE_BOT], axis = 0)  
    VE         = VE_VALS[VE_VALS.mask == False ].reshape(npanel,nalpha,nRe).data
    DVE        = DVE_VALS[DVE_VALS.mask == False ].reshape(npanel,nalpha,nRe).data    
    
    # ------------------------------------------------------------------------------------------------------
    # Compute effective surface of airfoil with boundary layer and recompute aerodynamic properties  
    # ------------------------------------------------------------------------------------------------------     
    DELTA             = np.nan_to_num(DELTA)
    new_y_coord       = np.zeros((npanel,nalpha,nRe)) 
    new_x_coord       = np.zeros((npanel,nalpha,nRe)) 
    new_y_coord       = Y + DELTA*normals[:,1,:,:]
    new_x_coord       = X + DELTA*normals[:,0,:,:]
    zeros             = np.zeros((nalpha,nRe))
    y_coord_3d_bl     = np.flip(np.insert(new_y_coord, int(npanel/2) ,zeros,axis = 0),axis = 0)
    x_coord_3d_bl     = np.flip(np.insert(new_x_coord, int(npanel/2) ,zeros,axis = 0),axis = 0) 
    x_coord_3d_bl     = x_coord_3d_bl - x_coord_3d_bl.min(axis=0) # shift airfoils so start value is 0 
    
    X_BL, Y_BL,vt_bl,cos_t_bl,normals_bl = hess_smith(x_coord_3d_bl,y_coord_3d_bl,alpha,Re_L,npanel,batch_analyis)      
    
    # ---------------------------------------------------------------------
    # Bottom surface of airfoil with boundary layer 
    # ---------------------------------------------------------------------     
    VT_BL              = np.ma.masked_greater(vt_bl,0 )
    VT_mask_BL         = np.ma.masked_greater(vt_bl,0 ).mask
    X_BOT_VALS_BL      = np.ma.array(X_BL, mask = VT_mask_BL)[::-1]
    Y_BOT_BL           = np.ma.array(Y_BL, mask = VT_mask_BL)[::-1]
         
    X_BOT_BL        = np.zeros_like(X_BOT_VALS_BL)
    X_BOT_BL[1:]    = np.cumsum(np.sqrt((X_BOT_VALS_BL[1:] - X_BOT_VALS_BL[:-1])**2 + (Y_BOT_BL[1:] - Y_BOT_BL[:-1])**2),axis = 0)
    first_idx       = np.ma.count_masked(X_BOT_BL,axis = 0)
    mask_count      = np.ma.count(X_BOT_BL,axis = 0)
    prev_index      = first_idx-1
    first_panel     = list(prev_index.flatten())
    last_panel      = list((first_idx-1 + mask_count).flatten())
    last_paneldve   = list((first_idx-2 + mask_count).flatten())
    aoas            = list(np.repeat(np.arange(nalpha),nRe))
    res             = list(np.tile(np.arange(nRe),nalpha) )
    X_BOT_BL.mask[first_panel,aoas,res] = False
    
    # flow velocity and pressure of on botton surface 
    VE_BOT_BL          = -VT_BL[::-1] 
    CP_BOT_BL          = 1 - VE_BOT_BL**2    
         
    # ---------------------------------------------------------------------
    # Top surface of airfoil with boundary layer 
    # ---------------------------------------------------------------------     
    VT_BL              = np.ma.masked_less(vt_bl,0 )
    VT_mask_BL         = np.ma.masked_less(vt_bl,0 ).mask
    X_TOP_VALS_BL      = np.ma.array(X_BL, mask = VT_mask) 
    Y_TOP_BL           = np.ma.array(Y_BL, mask = VT_mask) 

    X_TOP_BL        = np.zeros_like(X_TOP_VALS_BL)
    X_TOP_BL[1:]    = np.cumsum(np.sqrt((X_TOP_VALS_BL[1:] - X_TOP_VALS_BL[:-1])**2 + (Y_TOP_BL[1:] - Y_TOP_BL[:-1])**2),axis = 0)
    first_idx       = np.ma.count_masked(X_TOP_BL,axis = 0)
    mask_count      = np.ma.count(X_TOP_BL,axis = 0)
    prev_index      = first_idx-1
    first_panel     = list(prev_index.flatten())
    last_panel      = list((first_idx-1 + mask_count).flatten())
    last_paneldve   = list((first_idx-2 + mask_count).flatten())
    aoas            = list(np.repeat(np.arange(nalpha),nRe))
    res             = list(np.tile(np.arange(nRe),nalpha) )
    X_TOP_BL.mask[first_panel,aoas,res] = False

    # flow velocity and pressure of on botton surface 
    VE_TOP_BL          = VT_BL 
    CP_TOP_BL          = 1 - VE_TOP_BL**2 
    CP_VALS            = np.ma.concatenate([CP_BOT_BL,CP_TOP_BL], axis = 0 )  
    CP                 = CP_VALS[CP_VALS.mask == False ].reshape(npanel,nalpha,nRe).data
    AERO_RES           = aero_coeff(X_BL,Y_BL,CP,alpha,npanel) 
    
    airfoil_properties = Data(
        AoA        = alpha,
        Re         = Re_L,
        Cl         = AERO_RES.Cl,
        Cd         = AERO_RES.Cd,
        Cm         = AERO_RES.Cm,
        normals    = normals,
        x          = X,
        y          = Y,        
        x_bl       = X_BL,
        y_bl       = Y_BL,
        Cp         = CP,         
        Ue_Vinf    = VE,         
        dVe        = DVE,   
        theta      = THETA,      
        delta_star = DELTA_STAR, 
        delta      = DELTA,
        Re_theta   = RE_THETA,
        Re_x       = RE_X,
        H          = H,               
        Cf         = CF,          
        )  
    
    if batch_analyis:
        pass
    else:
        airfoil_properties = extract_values(airfoil_properties) 
        
    return  airfoil_properties 


def interpolate_values(X_BOT,X_TOP,X_BOT_SURF,X_TOP_SURF,FUNC_BOT_SURF,FUNC_TOP_SURF,npanel,nalpha,nRe): 
    '''Interpolation of airfoil properties''' 
    FUNC = np.zeros((npanel,nalpha,nRe))
    for a_i in range(nalpha):
        for Re_i in range(nRe):
            top_func          = interp1d(np.flip(X_TOP_SURF, axis=0)[:,a_i,Re_i],np.flip(FUNC_TOP_SURF, axis=0)[:,a_i,Re_i], fill_value='extrapolate')
            bot_func          = interp1d(X_BOT_SURF[:,a_i,Re_i],FUNC_BOT_SURF[:,a_i,Re_i], fill_value='extrapolate')  
            x_top             = np.flip(X_TOP, axis=0)[:,a_i,Re_i]
            x_top             = x_top[x_top.mask == False].data    
            x_bot             = X_BOT[:,a_i,Re_i]
            x_bot             = x_bot[x_bot.mask == False].data                  
            FUNC[:,a_i,Re_i]  = np.concatenate([top_func(x_top),bot_func(x_bot)])
    return FUNC


def extract_values(AP):    
    AP.Cl         = np.diagonal(AP.Cl,axis1 = 0, axis2 = 1)
    AP.Cd         = np.diagonal(AP.Cd,axis1 = 0, axis2 = 1)
    AP.Cm         = np.diagonal(AP.Cm,axis1 = 0, axis2 = 1)
    AP.normals    = np.diagonal(AP.normals,axis1 = 2, axis2 = 3)
    AP.x          = np.diagonal(AP.x,axis1 = 1, axis2 = 2)
    AP.y          = np.diagonal(AP.y,axis1 = 1, axis2 = 2)
    AP.x_bl       = np.diagonal(AP.x_bl,axis1 = 1, axis2 = 2)
    AP.y_bl       = np.diagonal(AP.y_bl,axis1 = 1, axis2 = 2)
    AP.Cp         = np.diagonal(AP.Cp,axis1 = 1, axis2 = 2)         
    AP.Ue_Vinf    = np.diagonal(AP.Ue_Vinf,axis1 = 1, axis2 = 2)        
    AP.dVe        = np.diagonal(AP.dVe,axis1 = 1, axis2 = 2)   
    AP.theta      = np.diagonal(AP.theta,axis1 = 1, axis2 = 2)      
    AP.delta_star = np.diagonal(AP.delta_star,axis1 = 1, axis2 = 2)
    AP.delta      = np.diagonal(AP.delta,axis1 = 1, axis2 = 2)
    AP.Re_theta   = np.diagonal(AP.Re_theta,axis1 = 1, axis2 = 2)
    AP.Re_x       = np.diagonal(AP.Re_x,axis1 = 1, axis2 = 2)
    AP.H          = np.diagonal(AP.H,axis1 = 1, axis2 = 2)                
    AP.Cf         = np.diagonal(AP.Cf,axis1 = 1, axis2 = 2)
    
    return AP
    
    
    