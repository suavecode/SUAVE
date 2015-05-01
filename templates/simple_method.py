# simple_method.py
# 
# Created:  Jan 2015, J. Dawson
# Modified: 

## style note --
## this is a stand alone method, and should be lower_case_with_underscore

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
# these should start with SUAVE, unless importing locally
from SUAVE.Core import (
    Data, Container, Data_Exception, Data_Warning,
)

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np
import scipy as sp
# import pylab as plt


# ----------------------------------------------------------------------
#  Simple Method
# ----------------------------------------------------------------------

def simple_method(input1,input2=None):
    """ SUAVE.Methods.SimpleMethod(input1,input2=None)
        does something useful
        
        Inputs:
            input1 - description [units]
            input2 - description [units]
            
        Outputs:
            output1 - description
            output2 - description
            >> try to minimize outputs
            >> pack up outputs into Data() if needed
        
        Assumptions:
            if needed
        
    """
    
    # unpack inputs
    var1 = input1.var1
    var2 = inputs.var2
    
    # setup
    var3 = var1 * var2
    
    # process
    magic = np.log(var3)
    
    # packup outputs
    output = Data()
    output.magic = magic
    output.var3  = var3
    
    return output
    
        
# ----------------------------------------------------------------------
#   Helper Functions
# ----------------------------------------------------------------------
# these will not be available in the SUAVE namespace

def helper_function(input1,inputs2=None):
    """ conditions data for some useful purpose
        
        Inputs:
            input1 - description [units]
            input2 - description [units]
            
        Outpus:
            output1 - description
            output2 - description
            >> try to minimize outputs
            >> pack up outputs into Data() if needed
        
        Assumptions:
            if needed
    """
    
    # unpack inputs
    var1 = input1.var1
    var2 = inputs.var2
    
    # setup
    var3 = var1 * var2
    
    # process
    magic = np.log(var3)
    
    # packup outputs
    output = Data()
    output.magic = magic
    output.var3  = var3
    
    return output
    
    
# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':    
    raise RuntimeError , 'test failed, not implemented'