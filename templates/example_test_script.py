
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

def main(block_plot=True):
    # this is the function that will be called by automatic_regression.py
    # the block_plot flag is needed to let the script continue after 
    # drawing plots
    
    ## do the test
    a = 5 + 2
    
    ## check results
    # raise an Exception if something doesn't work out
    # this will be caught by the automatic regression script and logged 
    # appropriately
    test = 'good'
    if not test == 'good':
        raise Exception

    ## plotting
    # if this is needed, make sure to let the automatic regression script 
    # to keep working by leaving a flag to not wait (block) to view plots
    plt.show(block=block_plot)
    plt.close()
 
    return


# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
