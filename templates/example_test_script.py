# example_test_script.py
# 
# Created:  Jan 2015, J. Dawson
# Modified: 

## style note --
## this is a test script that exercises the code.

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import SUAVE
from SUAVE.Core import Units

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------        
#   Main
# ----------------------------------------------------------------------    

def main():
    """ this is the function that will be called by automatic_regression.py """
    
    # ------------------------------------------------------------------
    #   The Tests
    # ------------------------------------------------------------------
    
    a = 5 + 2
    test = 'good'
    
    
    # ------------------------------------------------------------------
    #   Plotting
    # ------------------------------------------------------------------    
    
    # if needed
    plt.plot([1,2,3],[0,-1,2],'b-')
    
    ## IMPORTANT
    # do not include plt.show() in this function
    # it will prevent the regression script from continuing    

    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------
    
    # raise an Exception if something doesn't work out
    # this will be caught by the automatic regression script and logged 
    # appropriately
    assert( test == 'good' )


    return

#: def main()


# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
    
    # you can call plt.show() here
    plt.show()