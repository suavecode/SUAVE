## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
# post_processing.py
# 
# Created:  Aug 2022, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
import SUAVE 
import numpy as np      
from SUAVE.Methods.Aerodynamics.Airfoil_Panel_Method.supporting_functions import *

#-------------------------------------------------------------------------------
def calc_force(Airfoils):
    """Calculates force and moment coefficients
    
    Assumptions:
      lift/moment are computed from a panel pressure integration
      the cp distribution is stored as well

    Source:   
                                                     
    Inputs:
       Airfoils :  structure with solution (inviscid or viscous)
                                                                           
    Outputs   
       Airfoils.post values are filled in
    
    Properties Used:
    N/A
    """     

    chord = Airfoils.geom.chord 
    xref  = Airfoils.geom.xref # chord and ref moment point 
    Vinf  = Airfoils.param.Vinf 
    rho   = Airfoils.oper.rho 
    alpha = Airfoils.oper.alpha
    qinf  = 0.5*rho*Vinf**2 # dynamic pressure
    N     = Airfoils.foil.N # number of points on the airfoil

    # calculate the pressure coefficient at each node
    if Airfoils.oper.viscous:
        ue = np.atleast_2d(Airfoils.glob.U[3,:]).T
    else:
        ue = get_ueinv(Airfoils) 
    cp, cp_ue    = get_cp(ue, Airfoils.param) 
    Airfoils.post.cp    = cp
    Airfoils.post.cpi,_ = get_cp(get_ueinv(Airfoils),Airfoils.param) # inviscid cp # lift, moment, near-field pressure cd coefficients by cp integration   
    cl                  = 0 
    cl_ue               = np.zeros((N,1))
    cl_alpha            = 0
    cm                  = 0
    cdpi                = 0  
    for i0 in range(1,N+1):
        i  = i0
        ip = i-1 
        if (i0==N):
            i  = 0 
            ip = N-1 
        x1       = Airfoils.foil.x[:,ip] 
        x2       = Airfoils.foil.x[:,i] # panel points
        dxv      = x2-x1 
        dx1      = x1-xref
        dx2      = x2-xref
        dx1nds   = dxv[0]*dx1[0]+dxv[1]*dx1[1] # (x1-xref) cross n*ds
        dx2nds   = dxv[0]*dx2[0]+dxv[1]*dx2[1] # (x2-xref) cross n*ds
        dx       = -dxv[0]*np.cos(alpha) - dxv[1]*np.sin(alpha) # minus from CW node ordering
        dz       =  dxv[1]*np.cos(alpha) - dxv[0]*np.sin(alpha) # for drag
        cp1      = cp[ip] 
        cp2      = cp[i]
        cpbar    = 0.5*(cp[ip]+cp[i]) # average cp on the panel
        cl       = cl + dx*cpbar
        I        = [ip,i]  
        cl_ue[I,0] = cl_ue[I,0] + dx*0.5*cp_ue[I,0]
        cl_alpha = cl_alpha + cpbar*(np.sin(alpha)*dxv[0] - np.cos(alpha)*dxv[1])*np.pi/180
        cm       = cm + cp1*dx1nds/3 + cp1*dx2nds/6 + cp2*dx1nds/6 + cp2*dx2nds/3
        cdpi     = cdpi + dz*cpbar
       
    Airfoils.post.cl       = cl/chord 
    Airfoils.post.cl_ue    = cl_ue 
    Airfoils.post.cl_alpha = cl_alpha
    Airfoils.post.cm       = cm/chord**2 
    Airfoils.post.cdpi     = cdpi/chord
  
    # viscous contributions
    cd  = np.array([0])
    cdf = np.array([0])
    if Airfoils.oper.viscous:
  
        # Squire-Young relation for total drag (exrapolates theta from  of wake)
        iw = Airfoils.vsol.Is[2][-1] # station at the  of the wake
        U  = Airfoils.glob.U[:,iw] 
        H  = U[1]/U[0] 
        ue, _ = get_uk(U[3], Airfoils.param) # state
        cd = np.array([2.0])*U[0]*(ue/Vinf)**((5+H)/2.)
    
        # skin friction drag
        Df = np.array([0.])
        for iss in range(2):
            Is    = Airfoils.vsol.Is[iss] # surface point indices
            param = build_param(Airfoils, iss) # get parameter structure
            param = station_param(Airfoils, param, Is[0])
            cf1   = 0  # first cf value
            ue1   = 0  
            rho1  = rho
            x1 = Airfoils.isol.xstag
            for i in range(len(Is)): # loop over points
                param = station_param(Airfoils, param, Is[i])
                cf2,_   = get_cf(Airfoils.glob.U[:,Is[i]], param) # get cf value
                ue2,_   = get_uk(Airfoils.glob.U[3,Is[i]], param)
                rho2,_  = get_rho(Airfoils.glob.U[:,Is[i]], param)
                x2    = np.atleast_2d(Airfoils.foil.x[:,Is[i]]).T
                dxv   = x2 - x1
                dx    = dxv[0]*np.cos(alpha) + dxv[1]*np.sin(alpha)
                Df    = Df + 0.25*(rho1*cf1*ue1**2 + rho2*cf2*ue2**2)*dx
                cf1   = cf2 
                ue1   = ue2 
                x1    = x2
                rho1  = rho2 
        
        cdf = Df/(qinf*chord)    
      
    # store results
    Airfoils.post.cd  = cd 
    Airfoils.post.cdf = cdf
    Airfoils.post.cdp = cd-cdf 
    
    return  

#------------------------------------------------------------------------------- 
def get_distributions(Airfoils):
    """Computes various distributions (quantities at nodes) and stores them in Airfoils.post
    
    Assumptions:
    None

    Source:   
                                                     
    Inputs:
       Airfoils  : data class with a valid solution in Airfoils.glob.U
                                                                           
    Outputs   
       Airfoils.post : distribution quantities calculated
    
    Properties Used:
    N/A
    """    
    
    if np.shape(Airfoils.glob.U) != 0: 
        assert('no global solution')

    # quantities already in the global state
    Airfoils.post.theta      = Airfoils.glob.U[0,:] # theta
    Airfoils.post.delta_star = Airfoils.glob.U[1,:] # delta*
    Airfoils.post.sa         = Airfoils.glob.U[2,:] # amp or ctau
    Airfoils.post.ue , _     = get_uk(Airfoils.glob.U[3,:], Airfoils.param) # compressible edge velocity 
    Airfoils.post.uei        = get_ueinv(Airfoils) # compressible inviscid edge velocity

    # derived viscous quantities
    N     = Airfoils.glob.Nsys 
    cf    = np.zeros((N,1)) 
    Ret   = np.zeros((N,1))
    Hk    = np.zeros((N,1))
    de    = np.zeros((N,1))
    for iss in range(3):   # loop over surfaces
        Is    = Airfoils.vsol.Is[iss] # surface point indices
        param = build_param(Airfoils, iss) # get parameter structure
        for i in range(len(Is)):  # loop over points
            j         = Is[i]
            Uj        = Airfoils.glob.U[:,j]
            param     = station_param(Airfoils, param, j)
            uk ,_     = get_uk(Uj[3], param) # corrected edge speed
            cfloc, _  = get_cf(Uj, param) # local skin friction coefficient
            de[j],_   = get_de(Uj, param)
            cf[j]     = cfloc * uk**2/param.Vinf**2 # free-stream-based cf
            Ret[j],_  = get_Ret(Uj, param) # Re_theta
            Hk[j],_   = get_Hk(Uj, param) # kinematic shape factor 
    
    Airfoils.post.cf    = cf 
    Airfoils.post.Ret   = Ret 
    Airfoils.post.Hk    = Hk 
    Airfoils.post.delta = de
    
    # normals 
    t                     = np.concatenate((Airfoils.foil.t,Airfoils.wake.t),axis = 1)
    vec                   = np.concatenate((-t[1,:][None,:],t[0,:][None,:]) ,axis = 0)
    e                     = np.linalg.norm(vec,axis = 0)
    n                     = vec/np.tile(e[None,:], (2,1)) # outward normals   
    Airfoils.post.normals = n

    return 




