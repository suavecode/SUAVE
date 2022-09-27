## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
# supporting_functions.py
# 
# Created:  Aug 2022, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
import SUAVE  
import numpy as np     

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def build_param(Airfoils, iss):   
    """Builds a parameter structure for side iss
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
      iss  : side number, 0 = lower, 1 = upper, 2 = wake
                                                                           
    Outputs: 
      param : Airfoils.param structure with side information
    
    Properties Used:
    N/A
    """ 
    param      = Airfoils.param  
    param.wake = (iss == 2)
    param.turb = param.wake # the wake is fully turbulent
    param.simi = False # True for similarity station

    return param

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def station_param(Airfoils, param, i): 
    """Modifies parameter structure to be specific for station i
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
       i  : station number (node index along the surface)
                                                                           
    Outputs: 
       param : modified parameter structure
       
    Properties Used:
    N/A
    """  
    #   param : modified parameter structure
    param.turb = Airfoils.vsol.turb[i][0] # turbulent
    param.simi = i in Airfoils.isol.Istag # similarity

    return param


## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def get_ueinv(Airfoils):  
    """Computes invicid tangential velocity at every node
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs:
       Airfoils : data structure
                                                                           
    Outputs: 
       ueinv : inviscid velocity at airfoil and wake (if exists) points
    
    Properties Used:
    N/A
    """   
    alpha = Airfoils.oper.alpha  
    cs    = np.zeros((2,1))
    cs[0] = np.cos(alpha)
    cs[1] = np.sin(alpha) 
    uea   = (Airfoils.isol.sgnue*(np.matmul(Airfoils.isol.gamref,cs)).T ).T # airfoil
    if (Airfoils.oper.viscous) and (Airfoils.wake.N > 0):
        uew    = np.matmul(Airfoils.isol.uewiref,cs) # wake
        uew[0] = uea[-1] # ensures continuity of upper surface and wake ue
    else:
        uew = np.empty(shape=[0,1])
     
    ueinv= np.concatenate((uea, uew), axis = 0)# airfoil/wake edge velocity

    return ueinv

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def get_cp(u, param): 
    """Calculates pressure coefficient from speed, with compressibility correction
    
    Assumptions:
    Karman-Tsien correction is included

    Source:   
                                                     
    Inputs: 
      u     : speed
      param : parameter structure
                                                                           
    Outputs: 
      cp, cp_U : pressure coefficient and its linearization w.r.t. u
      
    Properties Used:
    N/A
    """        

    Vinf = param.Vinf
    cp = 1-(u/Vinf)**2 
    cp_u = -2*u/Vinf**2
    if (param.Minf > 0):
        l = param.KTl 
        b = param.KTb
        den = b+0.5*l*(1+b)*cp 
        den_cp = 0.5*l*(1+b)
        cp = cp/den 
        cp_u = cp_u * (1-cp*den_cp)/den
    return cp, cp_u 


## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def init_thermo(Airfoils):  
    """Initializes thermodynamics variables in param structure
    
    Assumptions:
       None

    Source:   
                                                     
    Inputs: 
      Airfoils  : data class with oper structure set
                                                                           
    Outputs: 
      Airfoils.param fields filled in based on Airfoils.oper
      Gets ready for compressibilty corrections if Airfoils.oper.Ma > 0
    
    Properties Used:
    N/A
    """ 

    g             = Airfoils.param.gam 
    gmi           = g-1
    rhoinf        = Airfoils.oper.rho # freestream density
    Vinf          = Airfoils.oper.Vinf 
    Airfoils.param.Vinf  = Vinf # freestream speed
    Airfoils.param.muinf = rhoinf*Vinf*Airfoils.geom.chord/Airfoils.oper.Re # freestream dyn viscosity 
    Minf          = Airfoils.oper.Ma 
    Airfoils.param.Minf  = Minf # freestream Mach
    if (Minf > 0):
        Airfoils.param.KTb = np.sqrt(1-Minf**2) # Karman-Tsien beta
        Airfoils.param.KTl = Minf**2/(1+Airfoils.param.KTb)**2 # Karman-Tsien lambda
        Airfoils.param.H0  = (1+0.5*gmi*Minf**2)*Vinf**2/(gmi*Minf**2) # stagnation enthalpy
        Tr          = 1-0.5*Vinf**2/Airfoils.param.H0 # freestream/stagnation temperature ratio
        finf        = Tr**1.5*(1+Airfoils.param.Tsrat)/(Tr + Airfoils.param.Tsrat) # Sutherland's ratio
        Airfoils.param.cps = 2/(g*Minf**2)*(((1+0.5*gmi*Minf**2)/(1+0.5*gmi))**(g/gmi) - 1)
    else:
        finf = 1 # incompressible case
      
    Airfoils.param.mu0  = Airfoils.param.muinf/finf  # stag visc (Sutherland ref temp is stag)
    Airfoils.param.rho0 = rhoinf*(1+0.5*gmi*Minf**2)**(1/gmi) # stag density 
    return 

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def get_upw(U1, U2, param):
    """ Calculates a local upwind factor (0.5 = trap 1 = BE) based on two states 
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
      U1,U2 : first/upwind and second/downwind states (4x1 each)
      param : parameter structure
                                                                           
    Outputs: 
      upw   : scalar upwind factor
      upw_U : 1x8 linearization vector, [upw_U1, upw_U2]
    
    Properties Used:
    N/A
    """     

    Hk1, Hk1_U1 = get_Hk(U1, param)
    Hk2, Hk2_U2 = get_Hk(U2, param)
    Z     = np.zeros(len(Hk1_U1))
    Hut   = 1.0 # triggering constant for upwinding
    C     = 5.0 
    if (param.wake):
        C = 1.0 
    Huc   = C*Hut/Hk2**2 # only deps on U2
    Huc_U =  np.concatenate((Z, -2*Huc/Hk2*Hk2_U2),axis = 0)
    aa   = (Hk2-1)/(Hk1-1) 
    sga  = np.sign(aa)
    la   = np.log(sga*aa)
    la_U =  np.concatenate((-1/(Hk1-1)*Hk1_U1, 1/(Hk2-1)*Hk2_U2),axis = 0)
    Hls  = la**2 
    Hls_U = 2*la*la_U
    if (Hls > 15):
        Hls = 15 
        Hls_U = Hls_U*0 
    upw   = 1 - 0.5*np.exp(-Hls*Huc)
    upw_U = -0.5*np.exp(-Hls*Huc)*(-Hls_U*Huc-Hls*Huc_U)

    return upw, upw_U



## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def upwind(upw, upw_U, f1, f1_U1, f2, f2_U2): 
    """Calculates an upwind average (and derivatives) of two scalars
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
      upw, upw_U : upwind scalar and its linearization w.r.t. U1,U2
      f1, f1_U   : first scalar and its linearization w.r.t. U1
      f2, f2_U   : second scalar and its linearization w.r.t. U2
                                                                           
    Outputs: 
      f    : averaged scalar
      f_U  : linearization of f w.r.t. both states, [f_U1, f_U2]
    
    Properties Used:
    N/A
    """ 

    f   = (1-upw)*f1 + upw*f2
    f_U = (-upw_U)*f1 + upw_U*f2 + np.concatenate(((1-upw)*f1_U1, upw*f2_U2),axis=0)
    
    return f, f_U
 

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def slimit_Hkc(Hkc0):
    """Smooth limit of Hkc = def of Hk and Ret
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
       Hkc0 : baseline value of Hkc
                                                                           
    Outputs: 
       Hkc : smoothly limited value in defined range
       rd  : derivative of Hkc w.r.t. the input Hkc0
    
    Properties Used:
    N/A
    """ 
    
    Hl = .01
    Hh = .05
    if (Hkc0 < Hh):
        rn = (Hkc0-Hl)/(Hh-Hl) 
        rn_Hkc0 = 1/(Hh-Hl)
        if (rn<0):
            rn=0. 
            rn_Hkc0 = 0. 
        rf = 3*rn**2 - 2*rn**3 
        rf_rn = 6*rn - 6*rn**2
        Hkc = Hl + rf*(Hh-Hl) 
        rd = rf_rn*rn_Hkc0*(Hh-Hl)
    else:
        Hkc = Hkc0 
        rd = 1
    
    return Hkc, rd 

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def get_uq(ds, ds_U, cf, cf_U, Hk, Hk_U, Ret, Ret_U, param): 
    """Calculates the equilibrium 1/ue*due/dx
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
       ds, ds_U   : delta star and linearization (1x4)
       cf, cf_U   : skin friction and linearization (1x4)
       Hk, Hk_U   : kinematic shape parameter and linearization (1x4)
       Ret, Ret_U : theta Reynolds number and linearization (1x4)
       param      : parameter structure
                                                                           
    Outputs: 
       uq, uq_U   : equilibrium 1/ue*due/dx and linearization w.r.t. state (1x4)
    
    Properties Used:
    N/A
    """   

    beta = param.GB
    A = param.GA 
    C = param.GC
    if (param.wake):
        A = A*param.Dlr 
        C = 0 
    # limit Hk (TODO smooth/eliminate)
    if (param.wake) and (Hk < 1.00005):
        Hk = 1.00005 
        Hk_U = Hk_U*0 
    if (not param.wake) and (Hk < 1.05):
        Hk = 1.05 
        Hk_U = Hk_U*0 
    Hkc   = Hk - 1 - C/Ret
    Hkc_U = Hk_U + C/Ret**2*Ret_U 

    if (Hkc < .01):
        Hkc = .01 
        Hkc_U = Hkc_U*0. 
    ut   = 0.5*cf - (Hkc/(A*Hk))**2
    ut_U = 0.5*cf_U - 2*(Hkc/(A*Hk))*(Hkc_U/(A*Hk) - Hkc/(A*Hk**2)*Hk_U)
    uq   = ut/(beta*ds)
    uq_U = ut_U/(beta*ds) - uq/ds * ds_U

    return uq, uq_U


## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def get_cttr(U, param): 
    """Calculates root of the shear stress coefficient at transition.
    Used to initialize the first turb station after transition
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
      U     : state vector [th ds sa ue]
      param : parameter structure
                                                                           
    Outputs: 
      cttr, cttr_U : sqrt(shear stress coeff) and its lin w.r.t. U (1x4)
    
    Properties Used:
    N/A
    """   

    param.wake = False # transition happens just before the wake starts
    cteq, cteq_U = get_cteq(U, param)
    Hk, Hk_U = get_Hk(U, param)
    if (Hk < 1.05):
        Hk = 1.05 
        Hk_U = Hk_U*0 
    C      = param.CtauC 
    E      = param.CtauE
    c      = C*np.exp(-E/(Hk-1)) 
    c_U    = c*E/(Hk-1)**2*Hk_U
    cttr   = c*cteq 
    cttr_U = c_U*cteq + c*cteq_U

    return cttr, cttr_U


## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def get_cteq(U, param): 
    """Calculates root of the equilibrium shear stress coefficient using 
     equilibrium shear stress correlations
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
        U     : state vector [th ds sa ue]
        param : parameter structure
                                                                           
    Outputs: 
        cteq, cteq_U : sqrt(equilibrium shear stress) and its lin w.r.t. U (1x4)
    
    Properties Used:
    N/A
    """ 
    #   uses equilibrium shear stress correlations
    CC         = 0.5/(param.GA**2*param.GB) 
    C          = param.GC
    Hk, Hk_U   = get_Hk(U, param)
    Hs, Hs_U   = get_Hs(U, param)
    H, H_U     = get_H(U)
    Ret, Ret_U = get_Ret(U, param)
    Us, Us_U   = get_Us(U, param)
    if (param.wake):
        if (Hk < 1.00005):
            Hk = 1.00005 
            Hk_U = Hk_U*0 
        Hkc   = Hk - 1 
        Hkc_U = Hk_U
    else:
        if (Hk < 1.05):
            Hk = 1.05 
            Hk_U = Hk_U*0 
        Hkc   = Hk - 1 - C/Ret
        Hkc_U = Hk_U + C/Ret**2*Ret_U 
        if (Hkc < 0.01):
            Hkc = 0.01 
            Hkc_U = Hkc_U*0. 
    
    num    = CC*Hs*(Hk-1)*Hkc**2
    num_U  = CC*(Hs_U*(Hk-1)*Hkc**2 + Hs*Hk_U*Hkc**2 + Hs*(Hk-1)*2*Hkc*Hkc_U)
    den    = (1-Us)*H*Hk**2
    den_U  = (-Us_U)*H*Hk**2 + (1-Us)*H_U*Hk**2 + (1-Us)*H*2*Hk*Hk_U
    cteq   = np.sqrt(num/den)
    cteq_U = 0.5/cteq*(num_U/den - num/den**2*den_U)

    return cteq, cteq_U


## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def get_Hs(U, param): 
    """Calculates Hs = Hstar = K.E. shape parameter, from U 
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
       U     : state vector [th ds sa ue]
       param : parameter structure
                                                                           
    Outputs: 
       Hs, Hs_U : Hstar and its lin w.r.t. U (1x4)
    
    Properties Used:
    N/A
    """ 
    Hk, Hk_U = get_Hk(U, param)

    # limit Hk (TODO smooth/eliminate)
    if (param.wake) and (Hk < 1.00005):
        Hk = 1.00005 
        Hk_U = Hk_U*0 
    if (not param.wake) and (Hk < 1.05):
        Hk = 1.05 
        Hk_U = Hk_U*0 

    if (param.turb): # turbulent
        Hsmin = 1.5
        dHsinf = .015
        Ret, Ret_U = get_Ret(U, param)
        # limit Re_theta and depence
        Ho = 4 
        Ho_U = 0.
        if (Ret > 400):
            Ho = 3 + 400/Ret 
            Ho_U = -400/Ret**2*Ret_U 
        Reb = Ret 
        Reb_U = Ret_U
        if (Ret < 200):
            Reb = 200 
            Reb_U = Reb_U*0 
        if (Hk < Ho):  # attached branch
            Hr = (Ho-Hk)/(Ho-1)
            Hr_U = (Ho_U - Hk_U)/(Ho-1) - (Ho-Hk)/(Ho-1)**2*Ho_U
            aa = (2-Hsmin-4/Reb)*Hr**2
            aa_U = (4/Reb**2*Reb_U)*Hr**2 + (2-Hsmin-4/Reb)*2*Hr*Hr_U
            Hs = Hsmin + 4/Reb + aa * 1.5/(Hk+.5)
            Hs_U = -4/Reb**2*Reb_U + aa_U*1.5/(Hk+.5) - aa*1.5/(Hk+.5)**2*Hk_U
        else: # separated branch
            lrb   = np.log(Reb) 
            lrb_U = 1/Reb*Reb_U
            aa    = Hk - Ho + 4/lrb
            aa_U  = Hk_U - Ho_U - 4/lrb**2*lrb_U
            bb    = .007*lrb/aa**2 + dHsinf/Hk
            bb_U  = .007*(lrb_U/aa**2 - 2*lrb/aa**3*aa_U) - dHsinf/Hk**2*Hk_U
            Hs    = Hsmin + 4/Reb + (Hk-Ho)**2*bb
            Hs_U  = -4/Reb**2*Reb_U + 2*(Hk-Ho)*(Hk_U-Ho_U)*bb + (Hk-Ho)**2*bb_U
        
        # slight Mach number correction
        M2, M2_U = get_Mach2(U, param) # squared edge Mach number
        den      = 1+.014*M2 
        den_M2   = .014
        Hs       = (Hs+.028*M2)/den
        Hs_U     = (Hs_U+.028*M2_U)/den - Hs/den*den_M2*M2_U
    else: # laminar
        a = Hk-4.35
        if (Hk < 4.35):
            num   = .0111*a**2 - .0278*a**3
            Hs    = num/(Hk+1) + 1.528 - .0002*(a*Hk)**2
            Hs_Hk = (.0111*2*a - .0278*3*a**2)/(Hk+1) - num/(Hk+1)**2 - .0002*2*a*Hk*(Hk+a)
        else:
            Hs    = .015*a**2/Hk + 1.528
            Hs_Hk = .015*2*a/Hk - .015*a**2/Hk**2 
        Hs_U = Hs_Hk*Hk_U  

    return Hs, Hs_U
 

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def get_uk(u, param):
    """Calculates Karman-Tsien corrected speed
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
      u     : incompressible speed
      param : parameter structure
                                                                           
    Outputs: 
      uk, uk_u : compressible speed and its linearization w.r.t. u
    
    Properties Used:
    N/A
    """  

    if (param.Minf > 0):
        l     = param.KTl 
        Vinf  = param.Vinf
        den   = 1-l*(u/Vinf)**2
        den_u = -2*l*u/Vinf**2
        uk    = u*(1-l)/den 
        uk_u  = (1-l)/den - (uk/den)*den_u
    else: 
        uk   = u 
        uk_u = 1 
    return uk, uk_u


## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def  get_Mach2(U, param):
    """Calculates squared Mach number
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
      U     : state vector [th ds sa ue]
      param : parameter structure
                                                                           
    Outputs: 
      M2, M2_U : squared Mach number and its linearization w.r.t. U (1x4)
    
    Properties Used:
    N/A
    """    

    if (param.Minf > 0):
        H0         = param.H0 
        g          = param.gam
        uk, uk_u   = get_uk(U[3], param)
        c2         = (g-1)*(H0-0.5*uk**2) 
        c2_uk      = (g-1)*(-uk) # squared speed of sound
        M2         = uk**2/c2 
        M2_uk      = 2*uk/c2 - M2/c2*c2_uk
        M2_U       = np.array([0,0,0,M2_uk*uk_u])
    else:
        M2   = 0. 
        M2_U = np.zeros(4) 

    return M2, M2_U


## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def get_H(U): 

    """Calculates H = shape parameter = delta*/theta, from U
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
      U     : state vector [th ds sa ue]
      param : parameter structure
                                                                           
    Outputs: 
       H, H_U : shape parameter and its linearization w.r.t. U (1x4)
    
    Properties Used:
    N/A
    """ 

    H   = U[1]/U[0]
    H_U = np.array([-H/U[0], 1/U[0], 0, 0])

    return H, H_U


## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def get_Hw(U, wgap): 
    """ Calculates Hw = wake gap shape parameter = wgap/theta
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
      U    : state vector [th ds sa ue]
      wgap : wake gap
                                                                           
    Outputs: 
      Hw, Hw_U : wake gap shape parameter and its linearization w.r.t. U (1x4)
    
    Properties Used:
    N/A
    """           

    Hw   = wgap/U[0] # wgap/th
    Hw_U =  np.array([-Hw/U[0],0,0,0])
    return Hw, Hw_U



## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def get_Hk(U, param): 
    """Calculates Hk = kinematic shape parameter, from U
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
      U     : state vector [th ds sa ue]
      param : parameter structure
                                                                           
    Outputs: 
      Hk, Hk_U : kinematic shape parameter and its linearization w.r.t. U (1x4)
    
    Properties Used:
    N/A
    """  

    H, H_U = get_H(U)

    if (param.Minf > 0):
        M2, M2_U = get_Mach2(U, param) # squared edge Mach number
        den      = (1+0.113*M2) 
        den_M2   = 0.113
        Hk       = (H-0.29*M2)/den
        Hk_U     = (H_U-0.29*M2_U)/den - Hk/den*den_M2*M2_U
    else:
        Hk   = H 
        Hk_U = H_U
    
    return Hk, Hk_U


## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def get_Hss(U, param): 
    """Calculates Hss = density shape parameter, from U
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
       U     : state vector [th ds sa ue]
       param : parameter structure
                                                                           
    Outputs: 
       Hss, Hss_U : density shape parameter and its linearization w.r.t. U (1x4)
    
    Properties Used:
    N/A
    """  

    M2, M2_U = get_Mach2(U, param) # squared edge Mach number
    Hk, Hk_U = get_Hk(U,param)
    num      = 0.064/(Hk-0.8) + 0.251 
    num_U    = -.064/(Hk-0.8)**2*Hk_U
    Hss      = M2*num 
    Hss_U    = M2_U*num + M2*num_U
    
    return Hss, Hss_U


## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def  get_de(U, param):
    """ Calculates simplified BL thickness measure
    
    Assumptions:
    delta is delta* incremented with a weighted momentum thickness, theta
    The weight on theta deps on Hk, and there is an overall cap

    Source:   
                                                     
    Inputs: 
       U     : state vector [th ds sa ue]
       param : parameter structure
                                                                           
    Outputs: 
       de, de_U : BL thickness "delta" and its linearization w.r.t. U (1x4)
    
    Properties Used:
    N/A
    """     

    Hk, Hk_U = get_Hk(U, param)
    aa       = 3.15 + 1.72/(Hk-1) 
    aa_U     = -1.72/(Hk-1)**2*Hk_U
    de       = U[0]*aa + U[1]
    de_U     =  np.array([aa,1,0,0]) + U[0]*aa_U
    dmx      = 12.0
    if (de > dmx*U[0]): 
        de = dmx*U[0]
        de_U =  np.array([dmx,0,0,0])

    return de, de_U


## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def get_rho(U, param): 
    """Calculates the density (useful if compressible)
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
       U     : state vector [th ds sa ue]
       param : parameter structure
                                                                           
    Outputs: 
       rho, rho_U : density and linearization
    
    Properties Used:
    N/A
    """  

    if (param.Minf > 0):
        M2, M2_U = get_Mach2(U, param) # squared edge Mach number
        uk, uk_u = get_uk(U[3], param) # corrected speed
        H0       = param.H0 
        gmi      = param.gam-1
        den      = 1+0.5*gmi*M2 
        den_M2   = 0.5*gmi
        rho      = param.rho0/den**(1/gmi) 
        rho_U    = (-1/gmi)*rho/den*den_M2*M2_U
    else:
        rho   = param.rho0 
        rho_U = np.zeros((1,4))
    
    return rho, rho_U   

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def get_Ret(U, param):
    """ Calculates theta Reynolds number, Re_theta, from U
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
       U     : state vector [th ds sa ue]
       param : parameter structure
                                                                           
    Outputs:
      Ret, Ret_U : Reynolds number based on the momentum thickness, linearization
    
    Properties Used:
    N/A
    """ 

    if (param.Minf > 0):
        M2, M2_U = get_Mach2(U, param) # squared edge Mach number
        uk, uk_u = get_uk(U[3], param) # corrected speed
        H0     = param.H0 
        gmi    = param.gam-1 
        Ts     = param.Tsrat
        Tr     = 1-0.5*uk**2/H0 
        Tr_uk  = -uk/H0 # edge/stagnation temperature ratio
        f      = Tr**1.5*(1+Ts)/(Tr+Ts) 
        f_Tr   = 1.5*f/Tr-f/(Tr+Ts) # Sutherland's ratio
        mu     = param.mu0*f 
        mu_uk  = param.mu0*f_Tr*Tr_uk # local dynamic viscosity
        den    = 1+0.5*gmi*M2
        den_M2 = 0.5*gmi
        rho    = param.rho0/den**(1/gmi) 
        rho_U  = (-1/gmi)*rho/den*den_M2*M2_U # density
        Ret    = rho*uk*U[0]/mu
        Ret_U  = rho_U*uk*U[0]/mu + (rho*U[0]/mu-Ret/mu*mu_uk)* np.array([0,0,0,uk_u]) + rho*uk/mu* np.array([1,0,0,0])
    else:
        Ret    = param.rho0*U[0]*U[3]/param.mu0
        Ret_U  = np.array([U[3], 0, 0, U[0]])/param.mu0
    
    return Ret, Ret_U 



## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def get_cf(U, param): 
    """Calculates cf = skin friction coefficient, from U 
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
      U     : state vector [th ds sa ue]
      param : parameter structure
                                                                           
    Outputs: 
      cf, cf_U : skin friction coefficient and its linearization w.r.t. U (1x4)
    
    Properties Used:
    N/A
    """

    if (param.wake):
        cf   = np.array([0]) 
        cf_U = np.zeros(4) 
        return cf, cf_U  # np.zero cf in wake
    Hk, Hk_U   = get_Hk(U, param)
    Ret, Ret_U = get_Ret(U, param)

    # TODO: limit Hk

    if (param.turb): # turbulent cf
        M2, M2_U = get_Mach2(U, param) # squared edge Mach number
        Fc       = np.sqrt(1+0.5*(param.gam-1)*M2)
        Fc_U     = 0.5/Fc*0.5*(param.gam-1)*M2_U
        aa       = -1.33*Hk 
        aa_U     = -1.33*Hk_U  
        if (aa < -17):
            aa   = -20+3*np.exp((aa+17)/3)
            aa_U = (aa+20)/3*aa_U  # TODO: np.ping me  
        bb   = np.log(Ret/Fc) 
        bb_U = Ret_U/Ret - Fc_U/Fc
        if (bb < 3):
            bb   = np.array([3]) 
            bb_U = bb_U*0 
        bb    = bb/np.log(10) 
        bb_U  = bb_U/np.log(10)
        cc    = -1.74 - 0.31*Hk 
        cc_U  = -0.31*Hk_U
        dd    = np.tanh(4.0-Hk/0.875) 
        dd_U  = (1-dd**2)*(-Hk_U/0.875)
        cf0   = 0.3*np.exp(aa)*bb**cc
        cf0_U = cf0*aa_U + 0.3*np.exp(aa)*cc*bb**(cc-1)*bb_U + cf0*np.log(bb)*cc_U
        cf    = (cf0 + 1.1e-4*(dd-1))/Fc
        cf_U  = (cf0_U + 1.1e-4*dd_U)/Fc - cf/Fc*Fc_U
    else: # laminar cf
        if (Hk < 5.5):
            num    = .0727*(5.5-Hk)**3/(Hk+1) - .07
            num_Hk = .0727*(3*(5.5-Hk)**2/(Hk+1)*(-1) - (5.5-Hk)**3/(Hk+1)**2)
        else:
            num    = .015*(1-1/(Hk-4.5))**2 - .07
            num_Hk = .015*2*(1-1/(Hk-4.5))/(Hk-4.5)**2
        
        cf   = num/Ret
        cf_U = num_Hk/Ret*Hk_U - num/Ret**2*Ret_U       

    return cf, cf_U


## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def get_cfxt(U, x, param):
    """Calculates cf*x/theta from the state
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
       U     : state vector [th ds sa ue]
       x     : distance along wall (xi)
       param : parameter structure
                                                                           
    Outputs: 
       cfxt,  : the combination cf*x/theta (calls cf def)
       cfxt_U : linearization w.r.t. U (1x4)
       cfxt_x : linearization w.r.t x (scalar) 
    
    Properties Used:
    N/A
    """      

    cf, cf_U  = get_cf(U, param)
    cfxt      = cf*x/U[0] 
    cfxt_U    = cf_U*x/U[0]
    cfxt_U[0] = cfxt_U[0] - cfxt/U[0]
    cfxt_x    = cf/U[0]
     
    return cfxt, cfxt_U, cfxt_x



## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def get_cfutstag(U, param):
    """ Calculates cf*ue*theta, used in stagnation station calculations
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
      U     : state vector [th ds sa ue]
      param : parameter structure
                                                                           
    Outputs: 
      F, F_U : value and linearization of cf*ue*theta
    
    Properties Used:
    N/A
    """ 

    U[3]     = 0 
    Hk, Hk_U = get_Hk(U, param)

    if (Hk < 5.5): 
        num    = .0727*(5.5-Hk)**3/(Hk+1) - .07
        num_Hk = .0727*(3*(5.5-Hk)**2/(Hk+1)*(-1) - (5.5-Hk)**3/(Hk+1)**2)
    else:
        num    = .015*(1-1/(Hk-4.5))**2 - .07
        num_Hk = .015*2*(1-1/(Hk-4.5))/(Hk-4.5)**2
    
    nu  = param.mu0/param.rho0
    F   = nu*num
    F_U = nu*num_Hk*Hk_U 

    return F, F_U

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def get_cdutstag(U, param):
    """ Calculates cDi*ue*theta, used in stagnation station calculations
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
       U     : state vector [th ds sa ue]
       param : parameter structure
                                                                           
    Outputs: 
       D, D_U : value and linearization of cDi*ue*theta
    
    Properties Used:
    N/A
    """     

    U[3]     = 0 
    Hk, Hk_U = get_Hk(U, param)

    if (Hk<4):
        num = .00205*(4-Hk)**5.5 + .207
        num_Hk = .00205*5.5*(4-Hk)**4.5*(-1)
    else:
        Hk1 = Hk-4
        num = -.0016*Hk1**2/(1+.02*Hk1**2) + .207
        num_Hk = -.0016*(2*Hk1/(1+.02*Hk1**2) - Hk1**2/(1+.02*Hk1**2)**2*.02*2*Hk1)
    
    
    nu  = param.mu0/param.rho0
    D   = nu*num
    D_U = nu*num_Hk*Hk_U

    return D, D_U


## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def get_cDixt(U, x, param):
    """ Calculates cDi*x/theta from the state
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
       U     : state vector [th ds sa ue]
       x     : distance along wall (xi)
       param : parameter structure
                                                                           
    Outputs: 
       cDixt,  : the combination cDi*x/theta (calls cDi def)
       cDixt_U : linearization w.r.t. U (1x4)
       cDixt_x : linearization w.r.t x (scalar)  
    
    Properties Used:
    N/A
    """ 

    cDi, cDi_U = get_cDi(U, param)
    cDixt      = cDi*x/U[0] 
    cDixt_U    = cDi_U*x/U[0] 
    cDixt_U[0] = cDixt_U[0] - cDixt/U[0]
    cDixt_x    = cDi/U[0]

    return cDixt, cDixt_U, cDixt_x
 

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def get_cDi(U, param):
    """ Calculates cDi = dissipation def = 2*cD/H*, from the state
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
       U     : state vector [th ds sa ue]
       param : parameter structure
                                                                           
    Outputs: 
    
    Properties Used:
    N/A
    """    

    if (param.turb): # turbulent includes wake

        # initialize to 0 will add components that are needed
        cDi   = 0 
        cDi_U = np.zeros(4)

        if (not param.wake):
            # turbulent wall contribution (0 in the wake) 
            cDi0, cDi0_U = get_cDi_turbwall(U, param)
            cDi          = cDi + cDi0 
            cDi_U        = cDi_U + cDi0_U
            cDil, cDil_U = get_cDi_lam(U, param) # for max check
        else:
            turb_flag    = param.turb 
            cDil, cDil_U = get_cDi_lamwake(U, param) # for max check
            param.turb   = turb_flag

        # outer layer contribution
        cDi0, cDi0_U = get_cDi_outer(U, param)
        cDi          = cDi + cDi0 
        cDi_U        = cDi_U + cDi0_U

        # laminar stress contribution
        cDi0, cDi0_U = get_cDi_lamstress(U, param)
        cDi          = cDi + cDi0 
        cDi_U        = cDi_U + cDi0_U

        # maximum check
        if (cDil > cDi):
            cDi   = cDil
            cDi_U = cDil_U 

        # double dissipation in the wake
        if (param.wake):
            cDi   = 2*cDi 
            cDi_U = 2*cDi_U 
    else:
        # just laminar dissipation
        cDi, cDi_U = get_cDi_lam(U, param)
    

    return cDi, cDi_U


## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def get_cDi_turbwall(U, param): 
    """Calculates the turbulent wall contribution to cDi
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
       U     : state vector [th ds sa ue]
       param : parameter structure
                                                                           
    Outputs: 
       cDi, cDi_U : dissipation def and its linearization w.r.t. U (1x4)
    
    Properties Used:
    N/A
    """  

    cDi = 0 
    cDi_U = np.zeros((1,4)) 
    if (param.wake):
        return 

    # get cf, Hk, Hs, Us
    cf, cf_U   = get_cf(U, param)
    Hk, Hk_U   = get_Hk(U, param)
    Hs, Hs_U   = get_Hs(U, param)
    Us, Us_U   = get_Us(U, param)
    Ret, Ret_U = get_Ret(U, param)

    lr     = np.log(Ret) 
    lr_U   = Ret_U/Ret
    Hmin   = 1 + 2.1/lr 
    Hmin_U = -2.1/lr**2*lr_U
    aa     = np.tanh((Hk-1)/(Hmin-1)) 
    fac    = 0.5 + 0.5*aa
    fac_U  = 0.5*(1-aa**2)*(Hk_U/(Hmin-1)-(Hk-1)/(Hmin-1)**2*Hmin_U) 
    cDi    = 0.5*cf*Us*(2/Hs)*fac
    cDi_U  = cf_U*Us/Hs*fac + cf*Us_U/Hs*fac - cDi/Hs*Hs_U + cf*Us/Hs*fac_U 
    
    return cDi, cDi_U

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def get_cDi_lam(U, param): 
    """Calculates the laminar dissipation def cDi
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
       U     : state vector [th ds sa ue]
       param : parameter structure
                                                                           
    Outputs: 
       cDi, cDi_U : dissipation def and its linearization w.r.t. U (1x4)
    
    Properties Used:
    N/A
    """   

    # first get Hk and Ret
    Hk, Hk_U   = get_Hk(U, param)
    Ret, Ret_U = get_Ret(U, param)

    if (Hk<4):
        num = .00205*(4-Hk)**5.5 + .207
        num_Hk = .00205*5.5*(4-Hk)**4.5*(-1)
    else:
        Hk1 = Hk-4
        num = -.0016*Hk1**2/(1+.02*Hk1**2) + .207
        num_Hk = -.0016*(2*Hk1/(1+.02*Hk1**2) - Hk1**2/(1+.02*Hk1**2)**2*.02*2*Hk1)
    
    cDi = num/Ret
    cDi_U = num_Hk/Ret*Hk_U - num/Ret**2*Ret_U

    return cDi, cDi_U


## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def get_cDi_lamwake(U, param):
    
    """ Laminar wake dissipation def cDi
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
       U     : state vector [th ds sa ue]
       param : parameter structure
                                                                           
    Outputs: 
       cDi, cDi_U : dissipation def and its linearization w.r.t. U (1x4)
    
    Properties Used:
    N/A
    """       

    param.turb = False # force laminar

    # depencies
    Hk, Hk_U   = get_Hk(U, param)
    Hs, Hs_U   = get_Hs(U, param)
    Ret, Ret_U = get_Ret(U, param)
    HsRet      = Hs*Ret
    HsRet_U    = Hs_U*Ret + Hs*Ret_U

    num    = 2*1.1*(1-1/Hk)**2*(1/Hk)
    num_Hk = 2*1.1*(2*(1-1/Hk)*(1/Hk**2)*(1/Hk)+(1-1/Hk)**2*(-1/Hk**2))
    cDi    = num/HsRet
    cDi_U  = num_Hk*Hk_U/HsRet - num/HsRet**2*HsRet_U

    return cDi, cDi_U


## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def get_cDi_outer(U, param):
    """ Turbulent outer layer contribution to dissipation def cDi
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
       U     : state vector [th ds sa ue]
       param : parameter structure
                                                                           
    Outputs: 
       cDi, cDi_U : dissipation def and its linearization w.r.t. U (1x4)
    
    Properties Used:
    N/A
    """       

    if (not param.turb):
        cDi   = 0 
        cDi_U = np.zeros((4,1)) 
        return cDi, cDi_U   # for np.pinging

    # first get Hs, Us
    Hs, Hs_U = get_Hs(U, param)
    Us, Us_U = get_Us(U, param)

    # shear stress: note, state stores ct**.5
    ct    = U[2]**2 
    ct_U  =  np.array([0,0,2*U[2],0]) 
    cDi   = ct*(0.995-Us)*2/Hs
    cDi_U = ct_U*(0.995-Us)*2/Hs + ct*(-Us_U)*2/Hs - ct*(0.995-Us)*2/Hs**2*Hs_U 
   
    return cDi, cDi_U


## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def get_cDi_lamstress(U, param):  
    """ Laminar stress contribution to dissipation def cDi
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
       U     : state vector [th ds sa ue]
       param : parameter structure
                                                                           
    Outputs:
       cDi, cDi_U : dissipation def and its linearization w.r.t. U (1x4)
    
    Properties Used:
    N/A
    """    

    # first get Hs, Us, and Ret
    Hs, Hs_U    = get_Hs(U, param)
    Us, Us_U    = get_Us(U, param)
    Ret, Ret_U  = get_Ret(U, param)
    HsRet       = Hs*Ret
    HsRet_U     = Hs_U*Ret + Hs*Ret_U

    num    = 0.15*(0.995-Us)**2*2
    num_Us = 0.15*2*(0.995-Us)*(-1)*2
    cDi    = num/HsRet
    cDi_U  = num_Us*Us_U/HsRet - num/HsRet**2*HsRet_U

    return cDi, cDi_U


## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def get_Us(U, param):  
    """ Calculates the np.linalg.normalized wall slip velocity Us 
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
       U     : state vector [th ds sa ue]
       param : parameter structure
                                                                           
    Outputs: 
       Us, Us_U : np.linalg.normalized wall slip velocity and its linearization w.r.t. U (1x4)
    
    Properties Used:
    N/A
    """  
    
    Hs, Hs_U = get_Hs(U, param)
    Hk, Hk_U = get_Hk(U, param)
    H, H_U   = get_H(U)

    # limit Hk (TODO smooth/eliminate)
    if (param.wake) and (Hk < 1.00005):
        Hk   = 1.00005 
        Hk_U = Hk_U*0 
    if (not param.wake) and (Hk < 1.05):
        Hk   = 1.05 
        Hk_U = Hk_U*0 

    beta = param.GB 
    bi   = 1/beta
    Us   = 0.5*Hs*(1-bi*(Hk-1)/H)
    Us_U = 0.5*Hs_U*(1-bi*(Hk-1)/H) + 0.5*Hs*(-bi*(Hk_U)/H +bi*(Hk-1)/H**2*H_U)  
    # limits
    if (not param.wake) and (Us>0.95):
        Us   = 0.98    
        Us_U = Us_U*0 
    if ( param.wake and (Us>0.99995)):
        Us   = 0.99995 
        Us_U = Us_U*0   

    return Us, Us_U


## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def get_damp(U, param): 
    """ Calculates the amplification rate, dn/dx, used in predicting transition
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs: 
       U     : state vector [th ds sa ue]
       param : parameter structure
                                                                           
    Outputs: 
       damp, damp_U : amplification rate and its linearization w.r.t. U (1x4)
    
    Properties Used:
    N/A
    """        

    Hk, Hk_U   = get_Hk(U, param) 
    Ret, Ret_U = get_Ret(U, param)
    th         = U[0]

    # limit Hk (TODO smooth/eliminate)
    if (Hk < 1.05):
        Hk   = 1.05 
        Hk_U = Hk_U*0 

    Hmi    = 1/(Hk-1) 
    Hmi_U  = -Hmi**2*Hk_U
    aa     = 2.492*Hmi**0.43 
    aa_U   = 0.43*aa/Hmi*Hmi_U
    bb     = np.tanh(14*Hmi-9.24) 
    bb_U   = (1-bb**2)*14*Hmi_U
    lrc    = aa + 0.7*(bb+1)
    lrc_U  = aa_U + 0.7*bb_U
    lten   = np.log(10) 
    lr     = np.log(Ret)/lten
    lr_U   = (1/Ret)*Ret_U/lten
    dl     = .1  # changed from .08 to make smoother
    damp   = np.array([0])
    damp_U = np.zeros(len(U.T)) # default no amplification
    if (lr >= lrc-dl):
        rn   = (lr-(lrc-dl))/(2*dl) 
        rn_U = (lr_U - lrc_U)/(2*dl)
        if (rn >= 1):
            rf   = 1
            rf_U = np.zeros(len(U.T))
        else:
            rf   = 3*rn**2-2*rn**3 
            rf_U = (6*rn-6*rn**2)*rn_U
          
        ar     = 3.87*Hmi-2.52 
        ar_U   = 3.87*Hmi_U
        ex     = np.exp(-ar**2) 
        ex_U   = ex*(-2*ar*ar_U)
        da     = 0.028*(Hk-1)-0.0345*ex 
        da_U   = 0.028*Hk_U-0.0345*ex_U
        af     = -0.05+2.7*Hmi-5.5*Hmi**2+3*Hmi**3+0.1*np.exp(-20*Hmi)
        af_U   = (2.7-11*Hmi+9*Hmi**2-1*np.exp(-20*Hmi))*Hmi_U
        damp   = rf*af*da/th
        damp_U = (rf_U*af*da + rf*af_U*da + rf*af*da_U)/th - damp/th*np.array([1,0,0,0])
        

    # extra amplification to ensure dn/dx > 0 near ncrit
    ncrit  = param.ncrit 
    Cea    = np.array([5])
    nx     = Cea*(U[2]-ncrit) 
    nx_U   = Cea* np.array([0,0,1,0])
    eex    = 1+np.tanh(nx) 
    eex_U  = (1-np.tanh(nx)**2)*nx_U 
    ed     = eex*.001/th
    ed_U   = eex_U*.001/th - ed/th*np.array([1,0,0,0])
    damp   = damp + ed
    damp_U = damp_U + ed_U 
        
    return damp, damp_U