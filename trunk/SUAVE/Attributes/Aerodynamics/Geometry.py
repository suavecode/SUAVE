# Geometry.py
# 
# Created:  Trent, Anil, Tarik, Feb 2014
# Modified:         


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
from SUAVE.Core import \
    Data, Container, Data_Exception, Data_Warning

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np
import scipy as sp


# ----------------------------------------------------------------------
#  Attribute
# ----------------------------------------------------------------------

class Geometry(Data):
    """ SUAVE.Attributes.Aerodynamics.Geometry()
        geometry data for a aerodynamics evaluation
        
        Attributes:
            none
            
        Methods:
            none
            
        Assumptions:
            none
        
    """
    
    def __defaults__(self):
        pass
        
    def __check__(self):
        pass
    

    
# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':    
    raise RuntimeError , 'module test failed, not implemented'





