
# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import SUAVE
from SUAVE.Attributes import Units

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------        
#   Main
# ----------------------------------------------------------------------    

def main():
    """ this is the function that will be called by automatic_regression.py """
    
    ## do the test
    a = 5 + 2
    
    ## check results
    # raise an Exception if something doesn't work out
    # this will be caught by the automatic regression script and logged 
    # appropriately
    test = 'good'
    assert( test == 'good' )

    ## IMPORTANT
    # do not include plt.show() in this function
    # it will prevent the regression script from continuing
 
    return


# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
    
    # you can call plt.show() here
    ## plt.show()