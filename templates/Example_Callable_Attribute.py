# Example_Callable_Attribute.py
# 
# Created:  Jan 2015, J. Dawson
# Modified: 

## style note --
## since this is an Attribute class, it is Camel_Case_With_Underscore()
## as a special judicious case when used for Analysis objects, it can
## include methods that manage vehicle analysis.

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
## remove any unnecessary imports

# suave imports
## these should start with SUAVE, unless importing locally
from SUAVE.Core import (
    Function, Data, Container, Data_Exception, Data_Warning,
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
#  Attribute
# ----------------------------------------------------------------------

class Example_Callable_Attribute(Function):
    """ SUAVE.Methods.Example_Callable_Attribute()
        an example method with defaults, high level description
        does something useful when called
        
        This Class is Callable:
            Method(input1,input2=None)
        
        Callable Inputs:
            input1 - description [units]
            input2 - description [units]
            
        Callable Outputs:
            output1 - descriptioni [units]
        
        Attributes:
            area   - description [units]
            taper  - description [units]
        
        Sub-Methods:
            do_this(input1,input2)  <no description, document in function>

        Assumptions:
            if needed
        
    """
    
    def __defaults__(self):
        # default attributes, 
        self.area = None    # [units]
        self.taper = None   # [units]
        
    def __check__(self):
        # called after initialized data
        # use to check the data's fields, and modify as needed
        # will not recieve any inputs other than self
        
        # for example
        if self.taper == 10:
            self.area = 20
            
    def __call__(self,input1,input2=None):
        # the method used with the class is called like a function
        # document at class level
        
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
    
    def do_this(input1,input2=None):
        """ SUAVE.Attributes.Attribute.do_this(input1,input2=None)
            conditions data for some useful purpose
            
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
#   Unit Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':    
    raise RuntimeError , 'test failed, not implemented'