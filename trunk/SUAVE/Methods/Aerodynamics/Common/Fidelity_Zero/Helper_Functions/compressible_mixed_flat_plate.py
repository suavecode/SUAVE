## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Helper_Functions
# compressible_mixed_flat_plate.py
# 
# Created:  Aug 2014, T. MacDonald
# Modified: Jan 2016, E. Botero


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np


# ----------------------------------------------------------------------
#  Compressible Mixed Flat Plate
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Helper_Functions
def compressible_mixed_flat_plate(Re,Ma,Tc,xt):
    """Computes the coefficient of friction for a flat plate given the 
    input parameters. Also returns the correction terms used in the
    computation.

    Assumptions:
    Reynolds number between 10e5 and 10e9
    xt between 0 and 1

    Source:
    adg.stanford.edu (Stanford AA241 A/B Course Notes)

    Inputs:
    Re (Reynolds number)                                             [Unitless]
    Ma (Mach number)                                                 [Unitless]
    Tc (temperature)                                                 [K]
    xt (turbulent transition point as a proportion of chord length)  [Unitless]

    Outputs:
    cf_comp (coefficient of friction)                                [Unitless]
    k_comp (compressibility correction)                              [Unitless]
    k_reyn (Reynolds number correction)                              [Unitless]

    Properties Used:
    N/A
    """     
    
    if xt < 0.0 or xt > 1.0:
        raise ValueError("Turbulent transition must be between 0 and 1")
    
    #if np.any(Re > 10**9) or np.any(Re < 10**5):
        ##print 'Warning: Reynolds number outside expected range - in file compressible_mixed_flat_plate.py'
        #pass
    
    Rex = Re*xt
    Rex[Rex==0.0] = 0.0001

    theta = 0.671*xt/(Rex**0.5)
    xeff  = (27.78*theta*Re**0.2)**1.25
    Rext  = Re*(1-xt+xeff)
    
    cf_turb  = 0.455/(np.log10(Rext)**2.58)
    cf_lam   = 1.328/(Rex**0.5)
    
    if xt > 0.0:
        cf_start = 0.455/(np.log10(Re*xeff)**2.58)
    else:
        cf_start = 0.0
    
    cf_inc = cf_lam*xt + cf_turb*(1-xt+xeff) - cf_start*xeff
    
    # compressibility correction
    Tw = Tc * (1. + 0.178*Ma*Ma)
    Td = Tc * (1. + 0.035*Ma*Ma + 0.45*(Tw/Tc - 1.))
    k_comp = (Tc/Td) 
    
    # reynolds correction
    Rd_w   = Re * (Td/Tc)**1.5 * ( (Td+216.) / (Tc+216.) )
    k_reyn = (Re/Rd_w)**0.2
    
    # apply corrections
    cf_comp = cf_inc * k_comp * k_reyn
    
    return cf_comp, k_comp, k_reyn

  
# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':    
    
    validation_test = 0
    import pylab as plt
    
    if validation_test:
    
        Re = np.logspace(5,9,5)
        ii = 0
        xts = np.array([0.0,.1,.2,.3,.4,.5,.6,.8,1.0])
        cf_comp = np.zeros([9,5])
        k_comp = np.zeros_like(cf_comp)
        k_reyn = np.zeros_like(cf_comp)
        for xt in xts:
            (cf_comp[ii,:], k_comp[ii,:], k_reyn[ii,:]) = compressible_mixed_flat_plate(Re,0.0,216.0,xt)
            if ii == 0:
                (cf_comp_turb,k_comp_t,k_reyn_t) = compressible_turbulent_flat_plate(Re,0.0,216.0)
            ii = ii + 1
            
        cf_comp_empirical = np.array([[.0073,.0045,.0031,.0022,.0016],
                                      [.0074,.0044,.0029,.0020,.0015],
                                      [.0073,.0041,.0026,.0018,.0014],
                                      [.0070,.0038,.0024,.0017,.0013],
                                      [.0067,.0036,.0021,.0015,.0011],
                                      [.0064,.0033,.0020,.0013,.0009],
                                      [.0060,.0029,.0017,.0011,.0007],
                                      [.0052,.0022,.0011,.0006,.0004],
                                      [.0041,.0013,.0004,.0001,.00005]])
        
            
        plt.figure("Skin Friction Coefficient v. Reynolds Number")
        axes = plt.gca()    
        for i in range(len(cf_comp[:,0])):
            axes.semilogx(Re, cf_comp_empirical[i,:], 'ro-')
            axes.semilogx(Re, cf_comp[i,:], 'bo-')
            axes.semilogx(Re, cf_comp_turb, 'go-')
        axes.set_xlabel('Reynolds Number')
        axes.set_ylabel('Cf')
        axes.grid(True)  
        plt.show()

    else:
        
        Re = 10**7
        Ma = 2.0
        Tc = 216.0
        xt = 0.6
        (cf_comp, k_comp, k_reyn) = compressible_mixed_flat_plate(Re, Ma, Tc, xt)
        
    print(cf_comp)
    print(k_comp)
    print(k_reyn)    