#cryocooler_test.py
#Created : Nov 2021,   S.Claridge


import SUAVE

import numpy as np

from  SUAVE.Components.Energy.Cooling.Cryocooler import Cryocooler 

# ----------------------------------------------------------------------        
#   Main
# ----------------------------------------------------------------------  

def main():
    
    # ------------------------------------------------------------------
    #   The Tests
    # ------------------------------------------------------------------
    
    #List with each cooler type and their truth value for energy_calc(3, 77, 300)
    cooler_types = [['fps', 62.7483790002092], ['GM', 94.11764705882356] , ['sPT',282.2201317027282], ['dPT', 4905.968928863452]]

    
    
    #Check that each cooler type provides expected output
    for cooler_type in cooler_types:
        
        #create new Cryocooler
        test = Cryocooler()
        
        #set Croyocooler type from list
        test.cooler_type = cooler_type[0]
        
        truth_input_power = cooler_type[1]

        actual_input_power = test.energy_calc(3, 77, 300)
        
        # ------------------------------------------------------------------
        #   Check Results
        # ------------------------------------------------------------------ 
    
        error_input_power = np.abs(truth_input_power - actual_input_power)
        
        assert(error_input_power < 1e-6), "Cryocooler regression test failed for " + cooler_type[0] + " cryocooler model"

# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
