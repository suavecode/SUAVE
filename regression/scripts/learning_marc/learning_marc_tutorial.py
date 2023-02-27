# learning_marc_tutorial.py
# 
# Created: Feb 2023, M. Clarke

# ----------------------------------------------------------------------
#  Imports 
# ---------------------------------------------------------------------- 
import numpy as np 
from MARC.Input_Output.MARC.learning_MARC import learning_MARC
 
def main():
    """ This is a simple script that helps new users to get accustomed to modifying code, 
    updating the associated regresison and using GitHub. This float shoulf be incremented by
    one each time a user attempts this tutorials.     
    
    Assumptions:
       N/A

    Source:
       None
        
    Inputs:
        None

    Outputs: 
        count
    
    Properties Used:
        None
    """
    
    true_value     = np.array([1.0])
    computed_value = learning_MARC()
    diff_RPM       = np.abs(computed_value - true_value)
    
    print('Computed Value  : ' + str(computed_value))
    print('True Value      : ' + str(true_value))
    print('Regression Error: ' + str(diff_RPM))
    assert np.abs((computed_value - true_value)/true_value) < 1e-6   
    
    return

if __name__ == '__main__':
    main()