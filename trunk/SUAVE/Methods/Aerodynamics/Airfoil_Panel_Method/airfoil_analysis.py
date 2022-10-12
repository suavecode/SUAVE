## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
# airfoil_analysis.py

# Created:  Mar 2021, M. Clarke
# Modified: Sep 2022, M. Clarke

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
# airfoil_analysis.py
# ----------------------------------------------------------------------   

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def airfoil_analysis(airfoil_geometry,alpha,Re_L,airfoil_stations = [0],
                     initial_momentum_thickness=1E-5,tolerance = 1E0):
    """This computes the aerodynamic polars as well as the boundary layer properties of 
    an airfoil at a defined set of reynolds numbers and angle of attacks

    Assumptions:
    Michel Criteria used for transition

    Source:
    N/A

    Inputs: 
    airfoil_geometry   - airfoil geometry points                                                             [unitless]
    alpha              - angle of attacks                                                                    [radians]
    Re_L               - Reynolds numbers                                                                     [unitless]
    batch_analysis     - boolean : If True: the specified number of angle of attacks and Reynolds            [boolean]
                                  numbers are used to create a table of 2-D results for each combination
                                  Note: Can only accomodate one airfoil
                                  
                                  If False:The airfoils specified are run and corresponding angle of attacks 
                                  and Reynolds numbers
                                  Note: The number of airfoils, angle of attacks and reynolds numbers must 
                                  all the same dimension                     
    
    Outputs: 
    airfoil_properties.
        AoA            - angle of attack                                                   [radians
        Re             - Reynolds number                                                   [unitless]
        Cl             - lift coefficients                                                 [unitless]
        Cd             - drag coefficients                                                 [unitless]
        Cm             - moment coefficients                                               [unitless]
        normals        - surface normals of airfoil                                        [unitless]
        x              - x coordinate points on airfoil                                    [unitless]
        y              - y coordinate points on airfoil                                    [unitless]
        x_bl           - x coordinate points on airfoil adjusted to include boundary layer [unitless]
        y_bl           - y coordinate points on airfoil adjusted to include boundary layer [unitless]
        Cp             - pressure coefficient distribution                                 [unitless]
        Ue_Vinf        - ratio of boundary layer edge velocity to freestream               [unitless]
        dVe            - derivative of boundary layer velocity                             [m/s-m]
        theta          - momentum thickness                                                [m]
        delta_star     - displacement thickness                                            [m]
        delta          - boundary layer thickness                                          [m]
        H              - shape factor                                                      [unitless]
        Cf             - local skin friction coefficient                                   [unitless]
        Re_theta_t     - Reynolds Number as a function of theta transition location        [unitless]
        tr_crit        - critical transition criteria                                      [unitless]
                        
    Properties Used:
    N/A
    """    
    
    nalpha       = len(alpha[0,:])
    nRe          = len(Re_L[0,:]) 
    nRe_cpts     = len(Re_L) 
    nairfoil     = len(airfoil_stations)  
    ncases       = nalpha 
    ncpts        = nRe_cpts
    x_coord      = np.take(airfoil_geometry.x_coordinates,airfoil_stations,axis=0).T 
    y_coord      = np.take(airfoil_geometry.y_coordinates,airfoil_stations,axis=0).T 
    npanel       = len(x_coord)-1 
         
    if (nairfoil!=  nalpha) and (nairfoil!=  nRe):
        raise AssertionError('Number of angle of attacks and Reynolds numbers must be equal to the number of stations')      
    x_coord_3d = np.repeat(x_coord[:,:,np.newaxis],nRe_cpts, axis = 2) # number of points, number of cases, number of control points 
    y_coord_3d = np.repeat(y_coord[:,:,np.newaxis],nRe_cpts, axis = 2) # number of points, number of cases, number of control points 
        
    # Begin by solving for velocity distribution at airfoil surface using inviscid panel simulation
    # these are the locations (faces) where things are computed , len = n panel
    # dimension of vt = npanel x ncases x ncpts
    X,Y,vt,normals = hess_smith(x_coord_3d,y_coord_3d,alpha,Re_L,npanel)  
    
    # Reynolds number 
    RE_L_VALS = Re_L.T 
    
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
    aoas            = list(np.repeat(np.arange(ncases),ncpts))
    res             = list(np.tile(np.arange(ncpts),ncases) )
    X_BOT.mask[first_panel,aoas,res] = False
    
    # flow velocity and pressure of on botton surface 
    VE_BOT          = -VT[::-1]  
    
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
    BOT_T_RESULTS  = thwaites_method(npanel,ncases,ncpts, L_BOT , RE_L_VALS, X_BOT, VE_BOT, DVE_BOT,tolerance,
                                     THETA_0=initial_momentum_thickness) 
    X_T_BOT          = BOT_T_RESULTS.X_T      
    THETA_T_BOT      = BOT_T_RESULTS.THETA_T     
    DELTA_STAR_T_BOT = BOT_T_RESULTS.DELTA_STAR_T  
    H_T_BOT          = BOT_T_RESULTS.H_T         
    CF_T_BOT         = BOT_T_RESULTS.CF_T   
    RE_THETA_T_BOT   = BOT_T_RESULTS.RE_THETA_T    
    RE_X_T_BOT       = BOT_T_RESULTS.RE_X_T      
    DELTA_T_BOT      = BOT_T_RESULTS.DELTA_T      
     
    # transition location  
    TR_CRIT_BOT      = RE_THETA_T_BOT - 1.174*(1 + 224000/RE_X_T_BOT)*RE_X_T_BOT**0.46  
    CRITERION_BOT    = np.ma.masked_greater(TR_CRIT_BOT,0 )  
    mask_count       = np.ma.count(CRITERION_BOT,axis = 0)  
    mask_count[mask_count == npanel] = npanel-1 
    transition_panel = list(mask_count.flatten()) 
    aoas             = list(np.repeat(np.arange(ncases),ncpts))
    res              = list(np.tile(np.arange(ncpts),ncases))
        
    X_TR_BOT          = X_T_BOT[transition_panel,aoas,res].reshape(ncases,ncpts)
    DELTA_STAR_TR_BOT = DELTA_STAR_T_BOT[transition_panel,aoas,res].reshape(ncases,ncpts)
    THETA_TR_BOT      = THETA_T_BOT[transition_panel,aoas,res].reshape(ncases,ncpts)    
    DELTA_TR_BOT      = DELTA_T_BOT[transition_panel,aoas,res].reshape(ncases,ncpts) 
    
    TURBULENT_SURF    = L_BOT.data  - X_TR_BOT
    TURBULENT_COORD   = np.ma.masked_less(X_BOT.data  - X_TR_BOT,0) 
    
    # turbulent boundary layer properties using heads method 
    BOT_H_RESULTS     = heads_method(npanel,ncases,ncpts,DELTA_TR_BOT ,THETA_TR_BOT , DELTA_STAR_TR_BOT,
                                   TURBULENT_SURF, RE_L_VALS,TURBULENT_COORD, VE_BOT, DVE_BOT,tolerance)
    
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
    
    # Flatten array to extract values 
    X_BOT_SURF_1          = X_BOT_SURF.flatten('F')
    THETA_BOT_SURF_1      = THETA_BOT_SURF.flatten('F') 
    DELTA_STAR_BOT_SURF_1 = DELTA_STAR_BOT_SURF.flatten('F') 
    H_BOT_SURF_1          = H_BOT_SURF.flatten('F')
    CF_BOT_SURF_1         = CF_BOT_SURF.flatten('F')
    RE_THETA_BOT_SURF_1   = RE_THETA_BOT_SURF.flatten('F') 
    RE_X_BOT_SURF_1       = RE_X_BOT_SURF.flatten('F')
    DELTA_BOT_SURF_1      = DELTA_BOT_SURF.flatten('F')  
    
    # Extract values that are not masked 
    X_BOT_SURF_2           = X_BOT_SURF_1.data[~X_BOT_SURF_1.mask]
    THETA_BOT_SURF_2       = THETA_BOT_SURF_1.data[~THETA_BOT_SURF_1.mask] 
    DELTA_STAR_BOT_SURF_2  = DELTA_STAR_BOT_SURF_1.data[~DELTA_STAR_BOT_SURF_1.mask]
    H_BOT_SURF_2           = H_BOT_SURF_1.data[~H_BOT_SURF_1.mask]
    CF_BOT_SURF_2          = CF_BOT_SURF_1.data[~CF_BOT_SURF_1.mask]
    RE_THETA_BOT_SURF_2    = RE_THETA_BOT_SURF_1.data[~RE_THETA_BOT_SURF_1.mask] 
    RE_X_BOT_SURF_2        = RE_X_BOT_SURF_1.data[~RE_X_BOT_SURF_1.mask]
    DELTA_BOT_SURF_2       = DELTA_BOT_SURF_1.data[~DELTA_BOT_SURF_1.mask]
    
    X_BOT_SURF           = X_BOT_SURF_2.reshape((npanel,ncases,ncpts),order = 'F')  
    Y_BOT_SURF           = Y_BOT 
    THETA_BOT_SURF       = THETA_BOT_SURF_2.reshape((npanel,ncases,ncpts),order = 'F')  
    DELTA_STAR_BOT_SURF  = DELTA_STAR_BOT_SURF_2.reshape((npanel,ncases,ncpts),order = 'F')  
    H_BOT_SURF           = H_BOT_SURF_2.reshape((npanel,ncases,ncpts),order = 'F')
    CF_BOT_SURF          = CF_BOT_SURF_2.reshape((npanel,ncases,ncpts),order = 'F')
    RE_THETA_BOT_SURF    = RE_THETA_BOT_SURF_2.reshape((npanel,ncases,ncpts),order = 'F')  
    RE_X_BOT_SURF        = RE_X_BOT_SURF_2.reshape((npanel,ncases,ncpts),order = 'F')  
    DELTA_BOT_SURF       = DELTA_BOT_SURF_2.reshape((npanel,ncases,ncpts),order = 'F')          
    
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
    aoas            = list(np.repeat(np.arange(ncases),ncpts))
    res             = list(np.tile(np.arange(ncpts),ncases) )
    X_TOP.mask[first_panel,aoas,res] = False
    
    # flow velocity and pressure of on botton surface 
    VE_TOP          = VT    
    
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
    TOP_T_RESULTS    = thwaites_method(npanel,ncases,ncpts, L_TOP , RE_L_VALS,X_TOP,VE_TOP, DVE_TOP,tolerance,
                                     THETA_0=initial_momentum_thickness) 
    X_T_TOP          = TOP_T_RESULTS.X_T      
    THETA_T_TOP      = TOP_T_RESULTS.THETA_T     
    DELTA_STAR_T_TOP = TOP_T_RESULTS.DELTA_STAR_T  
    H_T_TOP          = TOP_T_RESULTS.H_T         
    CF_T_TOP         = TOP_T_RESULTS.CF_T      
    RE_THETA_T_TOP   = TOP_T_RESULTS.RE_THETA_T    
    RE_X_T_TOP       = TOP_T_RESULTS.RE_X_T      
    DELTA_T_TOP      = TOP_T_RESULTS.DELTA_T      

    # transition location  
    TR_CRIT_TOP       = RE_THETA_T_TOP - 1.174*(1 + 224000/RE_X_T_TOP)*(RE_X_T_TOP**0.46)
    CRITERION_TOP     = np.ma.masked_greater(TR_CRIT_TOP,0 ) 
    mask_count        = np.ma.count(CRITERION_TOP,axis = 0)  
    mask_count[mask_count == npanel] = npanel-1  
    transition_panel  = list(mask_count.flatten()) 
    aoas              = list(np.repeat(np.arange(ncases),ncpts))
    res               = list(np.tile(np.arange(ncpts),ncases) )
  
    X_TR_TOP          = X_T_TOP[transition_panel,aoas,res].reshape(ncases,ncpts)
    DELTA_STAR_TR_TOP = DELTA_STAR_T_TOP[transition_panel,aoas,res].reshape(ncases,ncpts)
    THETA_TR_TOP      = THETA_T_TOP[transition_panel,aoas,res].reshape(ncases,ncpts)    
    DELTA_TR_TOP      = DELTA_T_TOP[transition_panel,aoas,res].reshape(ncases,ncpts) 
   
    TURBULENT_SURF    = L_TOP.data  - X_TR_TOP
    TURBULENT_COORD   = np.ma.masked_less( X_TOP.data  - X_TR_TOP,0)

    # turbulent boundary layer properties using heads method 
    TOP_H_RESULTS     = heads_method(npanel,ncases,ncpts,DELTA_TR_TOP ,THETA_TR_TOP , DELTA_STAR_TR_TOP,
                                   TURBULENT_SURF, RE_L_VALS,TURBULENT_COORD, VE_TOP, DVE_TOP,tolerance)

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
    
    # Concatenate laminar and turbulent vectors
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
    
    # Flatten array to extract values 
    X_TOP_SURF_1          = X_TOP_SURF.flatten('F')
    THETA_TOP_SURF_1      = THETA_TOP_SURF.flatten('F') 
    DELTA_STAR_TOP_SURF_1 = DELTA_STAR_TOP_SURF.flatten('F') 
    H_TOP_SURF_1          = H_TOP_SURF.flatten('F')
    CF_TOP_SURF_1         = CF_TOP_SURF.flatten('F')
    RE_THETA_TOP_SURF_1   = RE_THETA_TOP_SURF.flatten('F') 
    RE_X_TOP_SURF_1       = RE_X_TOP_SURF.flatten('F')
    DELTA_TOP_SURF_1      = DELTA_TOP_SURF.flatten('F')  
    
    # Extract values that are not masked 
    X_TOP_SURF_2           = X_TOP_SURF_1.data[~X_TOP_SURF_1.mask]
    THETA_TOP_SURF_2       = THETA_TOP_SURF_1.data[~THETA_TOP_SURF_1.mask] 
    DELTA_STAR_TOP_SURF_2  = DELTA_STAR_TOP_SURF_1.data[~DELTA_STAR_TOP_SURF_1.mask]
    H_TOP_SURF_2           = H_TOP_SURF_1.data[~H_TOP_SURF_1.mask]
    CF_TOP_SURF_2          = CF_TOP_SURF_1.data[~CF_TOP_SURF_1.mask]
    RE_THETA_TOP_SURF_2    = RE_THETA_TOP_SURF_1.data[~RE_THETA_TOP_SURF_1.mask] 
    RE_X_TOP_SURF_2        = RE_X_TOP_SURF_1.data[~RE_X_TOP_SURF_1.mask]
    DELTA_TOP_SURF_2       = DELTA_TOP_SURF_1.data[~DELTA_TOP_SURF_1.mask]
    
    X_TOP_SURF           = X_TOP_SURF_2.reshape((npanel,ncases,ncpts),order = 'F') 
    Y_TOP_SURF           = Y_TOP 
    THETA_TOP_SURF       = THETA_TOP_SURF_2.reshape((npanel,ncases,ncpts),order = 'F')   
    DELTA_STAR_TOP_SURF  = DELTA_STAR_TOP_SURF_2.reshape((npanel,ncases,ncpts),order = 'F') 
    H_TOP_SURF           = H_TOP_SURF_2.reshape((npanel,ncases,ncpts),order = 'F')
    CF_TOP_SURF          = CF_TOP_SURF_2.reshape((npanel,ncases,ncpts),order = 'F')
    RE_THETA_TOP_SURF    = RE_THETA_TOP_SURF_2.reshape((npanel,ncases,ncpts),order = 'F')  
    RE_X_TOP_SURF        = RE_X_TOP_SURF_2.reshape((npanel,ncases,ncpts),order = 'F')  
    DELTA_TOP_SURF       = DELTA_TOP_SURF_2.reshape((npanel,ncases,ncpts),order = 'F')          

    
    # ------------------------------------------------------------------------------------------------------
    # concatenate lower and upper surfaces   
    # ------------------------------------------------------------------------------------------------------ 
    X_PANEL    = concatenate_surfaces(X_BOT,X_TOP,X_BOT_SURF,X_TOP_SURF,npanel,ncases,ncpts)
    Y_PANEL    = concatenate_surfaces(X_BOT,X_TOP,Y_BOT_SURF,Y_TOP_SURF,npanel,ncases,ncpts)
    THETA      = concatenate_surfaces(X_BOT,X_TOP,THETA_BOT_SURF,THETA_TOP_SURF,npanel,ncases,ncpts)
    DELTA_STAR = concatenate_surfaces(X_BOT,X_TOP,DELTA_STAR_BOT_SURF,DELTA_STAR_TOP_SURF,npanel,ncases,ncpts) 
    H          = concatenate_surfaces(X_BOT,X_TOP,H_BOT_SURF,H_TOP_SURF,npanel,ncases,ncpts)  
    CF         = concatenate_surfaces(X_BOT,X_TOP,CF_BOT_SURF,CF_TOP_SURF,npanel,ncases,ncpts) 
    RE_THETA   = concatenate_surfaces(X_BOT,X_TOP,RE_THETA_BOT_SURF,RE_THETA_TOP_SURF,npanel,ncases,ncpts)  
    RE_X       = concatenate_surfaces(X_BOT,X_TOP,RE_X_BOT_SURF,RE_X_TOP_SURF,npanel,ncases,ncpts) 
    DELTA      = concatenate_surfaces(X_BOT,X_TOP,DELTA_BOT_SURF,DELTA_TOP_SURF,npanel,ncases,ncpts)   
     
    VE_VALS    = np.ma.concatenate([np.flip(VE_BOT,axis = 0),VE_TOP ], axis = 0)
    DVE_VALS   = np.ma.concatenate([np.flip(DVE_BOT,axis = 0),DVE_TOP], axis = 0)    
    VE_VALS_1  = VE_VALS.flatten('F')
    DVE_VALS_1 = DVE_VALS.flatten('F')   
    VE_VALS_2  = VE_VALS_1.data[~VE_VALS_1.mask]
    DVE_VALS_2 = DVE_VALS_1.data[~DVE_VALS_1.mask]  
    VE         = VE_VALS_2.reshape((npanel,ncases,ncpts),order = 'F') 
    DVE        = DVE_VALS_2.reshape((npanel,ncases,ncpts),order = 'F')  
    
    # ------------------------------------------------------------------------------------------------------
    # Compute effective surface of airfoil with boundary layer and recompute aerodynamic properties  
    # ------------------------------------------------------------------------------------------------------   
    DELTA          = np.nan_to_num(DELTA) # make sure no nans   
    DELTA_PTS      = np.concatenate((DELTA,DELTA[-1][np.newaxis,:,:]),axis = 0)
    DELTA_PTS      = np.concatenate((DELTA[0][np.newaxis,:,:],DELTA_PTS),axis = 0)  
    NORMALS_PTS    = np.concatenate((normals,normals[-1][np.newaxis,:,:]),axis = 0)
    NORMALS_PTS    = np.concatenate((normals[0][np.newaxis,:,:],NORMALS_PTS),axis = 0) 
    POINT_NORMALS  = 0.5*(NORMALS_PTS[1:] + NORMALS_PTS[:-1])  
    POINT_BLS      = 0.5*(DELTA_PTS[1:] + DELTA_PTS[:-1])  
    y_coord_3d_bl  = y_coord_3d+ POINT_BLS*POINT_NORMALS[:,1,:,:]
    x_coord_3d_bl  = x_coord_3d+ POINT_BLS*POINT_NORMALS[:,0,:,:]   
    
    X_BL, Y_BL,vt_bl,normals_bl = hess_smith(x_coord_3d_bl,y_coord_3d_bl,alpha,Re_L,npanel)      
      
    # ---------------------------------------------------------------------
    # Bottom surface of airfoil with boundary layer 
    # ---------------------------------------------------------------------       
    VT_BL           = np.ma.masked_greater(vt_bl,0 )
    VT_BL_mask      = np.ma.masked_greater(vt_bl,0 ).mask
    X_BL_BOT_VALS   = np.ma.array(X_BL, mask = VT_mask)[::-1]
    Y_BL_BOT        = np.ma.array(Y_BL, mask = VT_mask)[::-1] 
    X_BL_BOT        = np.zeros_like(X_BL_BOT_VALS)
    X_BL_BOT[1:]    = np.cumsum(np.sqrt((X_BL_BOT_VALS[1:] - X_BL_BOT_VALS[:-1])**2 + (Y_BL_BOT[1:] - Y_BL_BOT[:-1])**2),axis = 0)
    first_idx       = np.ma.count_masked(X_BL_BOT,axis = 0)
    mask_count      = np.ma.count(X_BL_BOT,axis = 0)
    prev_index      = first_idx-1
    first_panel     = list(prev_index.flatten())
    last_panel      = list((first_idx-1 + mask_count).flatten())
    last_paneldve   = list((first_idx-2 + mask_count).flatten())
    aoas            = list(np.repeat(np.arange(ncases),ncpts))
    res             = list(np.tile(np.arange(ncpts),ncases) )
    X_BL_BOT.mask[first_panel,aoas,res] = False
    
    # flow velocity and pressure of on botton surface 
    VE_BL_BOT          = -VT_BL[::-1] 
    CP_BL_BOT          = 1 - VE_BL_BOT**2   
         
    # ---------------------------------------------------------------------
    # Top surface of airfoil with boundary layer 
    # ---------------------------------------------------------------------    
    VT_BL           = np.ma.masked_less(vt_bl,0 )
    VT_BL_mask      = np.ma.masked_less(vt_bl,0 ).mask
    X_BL_TOP_VALS   = np.ma.array(X_BL, mask = VT_BL_mask) 
    Y_BL_TOP        = np.ma.array(Y_BL, mask = VT_BL_mask)  
 
    X_BL_TOP        = np.zeros_like(X_BL_TOP_VALS)
    X_BL_TOP[1:]    = np.cumsum(np.sqrt((X_BL_TOP_VALS[1:] - X_BL_TOP_VALS[:-1])**2 + (Y_BL_TOP[1:] - Y_BL_TOP[:-1])**2),axis = 0)
    first_idx       = np.ma.count_masked(X_BL_TOP,axis = 0)
    mask_count      = np.ma.count(X_BL_TOP,axis = 0)
    prev_index      = first_idx-1
    first_panel     = list(prev_index.flatten())
    last_panel      = list((first_idx-1 + mask_count).flatten())
    last_paneldve   = list((first_idx-2 + mask_count).flatten())
    aoas            = list(np.repeat(np.arange(ncases),ncpts))
    res             = list(np.tile(np.arange(ncpts),ncases) )
    X_BL_TOP.mask[first_panel,aoas,res] = False
    
    # flow velocity and pressure of on botton surface 
    VE_BL_TOP    = VT_BL
    CP_BL_TOP    = 1 - VE_BL_TOP**2     
    
    CP_BL_VALS   = np.ma.concatenate([np.flip(CP_BL_BOT,axis = 0),CP_BL_TOP], axis = 0 )  
    CP_BL_VALS_1 = CP_BL_VALS.flatten('F')  
    CP_BL_VALS_2 = CP_BL_VALS_1.data[~CP_BL_VALS_1.mask] 
    CP_BL        = CP_BL_VALS_2.reshape((npanel,ncases,ncpts),order = 'F')    
    DCP_DX       = np.diff(CP_BL,axis=0)/ np.diff(X,axis=0) 
    
    AERO_RES_BL  = aero_coeff(X,Y,-CP_BL,alpha,npanel) 
    
    airfoil_properties = Data(
        AoA        = alpha,
        Re         = Re_L,
        cl         = AERO_RES_BL.Cl,
        cd         = AERO_RES_BL.Cd,
        cm         = AERO_RES_BL.Cm,  
        normals    = np.transpose(normals,(3,2,0,1)),
        x          = np.transpose(X,(2,1,0)),
        y          = np.transpose(Y,(2,1,0)),
        x_bl       = np.transpose(X_BL ,(2,1,0)),
        y_bl       = np.transpose(Y_BL ,(2,1,0)),
        cp         = np.transpose(CP_BL,(2,1,0)),  
        dcp_dx     = np.transpose(DCP_DX,(2,1,0)),            
        Ue_Vinf    = np.transpose(VE   ,(2,1,0)),         
        dVe        = np.transpose(DVE  ,(2,1,0)),   
        theta      = np.transpose(THETA,(2,1,0)),      
        delta_star = np.transpose(DELTA_STAR,(2,1,0)),  
        delta      = np.transpose(DELTA,(2,1,0)),  
        Re_theta   = np.transpose(RE_THETA,(2,1,0)),  
        Re_x       = np.transpose(RE_X,(2,1,0)),  
        H          = np.transpose(H,(2,1,0)),            
        cf         = np.transpose(CF,(2,1,0)),    
        )  
        
    return  airfoil_properties 


def concatenate_surfaces(X_BOT,X_TOP,FUNC_BOT_SURF,FUNC_TOP_SURF,npanel,ncases,ncpts): 
    '''Interpolation of airfoil properties   
    
    Assumptions:
    None

    Source:
    None                                                                    
                                                                   
    Inputs:                                    
    X_BOT          - bottom surface of airfoil                                     [unitless]
    X_TOP          - top surface of airfoil                                        [unitless]
    FUNC_BOT_SURF  - airfoil property computation discretization on bottom surface [multiple units]
    FUNC_TOP_SURF  - airfoil property computation discretization on top surface    [multiple units]
    npanel         - number of panels                                              [unitless]
    ncases         - number of angle of attacks                                    [unitless]
    ncpts            - number of Reynolds numbers                                    [unitless]
                                                                 
    Outputs:                                           
    FUNC           - airfoil property in user specified discretization on entire
                     surface of airfoil                                            [multiple units]
      
    Properties Used:
    N/A  
    '''  
    FUNC = np.zeros((npanel,ncases,ncpts))  
    
    for case in range(ncases):
        for cpt in range(ncpts):   
            top_func          = FUNC_TOP_SURF[:,case,cpt][X_TOP[:,case,cpt].mask == False] 
            bot_func          = FUNC_BOT_SURF[:,case,cpt][X_BOT[:,case,cpt].mask == False]                  
            FUNC[:,case,cpt]  = np.concatenate([bot_func[::-1],top_func])
    return FUNC
    