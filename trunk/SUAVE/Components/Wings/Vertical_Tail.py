# Vertical_Tail.py
#
# Created:  Feb 2014, T. Lukacyzk, T. Orra
# Modified: Feb 2016, T. MacDonald
#           May 2020, E. Botero


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUave imports
from .Wing import Wing
import numpy as np

# ----------------------------------------------------------------------
#  Attribute
# ----------------------------------------------------------------------

class Vertical_Tail(Wing):
    """This class is used to define vertical tails SUAVE

    Assumptions:
    None

    Source:
    N/A

    Inputs:
    None

    Outputs:
    None

    Properties Used:
    N/A
    """ 
    def __defaults__(self):
        """This sets the default for vertical tails in SUAVE.
    
        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        N/A
        """ 
        self.tag       = 'vertical_stabilizer'
        self.vertical  = True
        self.symmetric = False
        self.generative_design_max_per_vehicle = 2
        self.generative_design_char_min_bounds = [0,1.,0.001,0.1,0.001,-np.pi/4,0.7,-1.,-1.]   
        self.generative_design_char_max_bounds = [5.,np.inf,1.0,np.inf,np.pi/3,np.pi/4,1.,1.,1.]        


# ----------------------------------------------------------------------
#   Unit Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':
    raise RuntimeError('test failed, not implemented')