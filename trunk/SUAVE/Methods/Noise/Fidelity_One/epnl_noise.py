#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:     
#
# Author:      CARIDSIL
#
# Created:     21/07/2015
# Copyright:   (c) CARIDSIL 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np

def epnl_noise(PNLT):
    PNLT_max = np.max(PNLT)
    
    duration_correction = 10*np.log10(np.sum(10**(PNLT/10)))-PNLT_max-13
    
    EPNL=PNLT_max+duration_correction
    
    return (EPNL)    
    
    