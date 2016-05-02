# compressible_turbulent_flat_plate.py
# 
# Created:  Dec 2013, SUAVE Team
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np

# ----------------------------------------------------------------------
#  Compressible Turbulent Flat Plate
# ----------------------------------------------------------------------


def compressible_turbulent_flat_plate(Re,Ma,Tc):

    # incompressible skin friction coefficient
    cf_inc = 0.455/(np.log10(Re))**2.58
    
    # compressibility correction
    Tw = Tc * (1. + 0.178*Ma**2.)
    Td = Tc * (1. + 0.035*Ma**2. + 0.45*(Tw/Tc - 1.))
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
    
    (cf_comp, k_comp, k_reyn) = compressible_turbulent_flat_plate(1.0*10.0**7.0,0.0,216.0)
    
    print cf_comp
    print k_comp
    print k_reyn    
