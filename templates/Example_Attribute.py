## @ingroup templates
# Example_Attribute.py
# 
# Created:  Jan 2015, J. Dawson
# Modified:

## style note --
## since this is an Attribute class, it is Camel_Case_With_Underscore()
## it should not have any major analysis methods, 
## only data manipulation methods

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
## remove any unnecessary imports

# suave imports
## these should start with SUAVE, unless importing locally
from SUAVE.Core import (
    Data, Container
)

# python imports
import os, sys, shutil
from warnings import warn

# package imports
import numpy as np
import scipy as sp
# import pylab as plt


# ----------------------------------------------------------------------
#  Attribute
# ----------------------------------------------------------------------

## @ingroup templates
class Example_Attribute(Data):
    """<Description>
    
    Assumptions:
    <any assumptions>
    
    Source:
    <source>
    """
    
    def __defaults__(self):
        """<Description>
        
        Assumptions:
        <any assumptions>
        
        Source:
        <source>
        
        Inputs:
        <input1> <units>
        <input2> <units>
        ..
        
        Outputs:
        <output1> <units>
        <output2> <units>
        ..
        
        Properties Used:
        <property1> <units>
        <property2> <units>
        ..
        """        
        # default attributes, 
        self.area = None    # [units]
        self.taper = None   # [units]
        
        
    def __check__(self):
        """<Description>
        
        Assumptions:
        <any assumptions>
        
        Source:
        <source>
        
        Inputs:
        <input1> <units>
        <input2> <units>
        ..
        
        Outputs:
        <output1> <units>
        <output2> <units>
        ..
        
        Properties Used:
        <property1> <units>
        <property2> <units>
        ..
        """          
        # called after initialized data
        # use to check the data's fields, and modify as needed
        # will not recieve any inputs other than self
        
        # for example
        if self.taper == 10:
            self.area = 20
        
    def do_this(input1,input2=None):
        """<Description>
        
        Assumptions:
        <any assumptions>
        
        Source:
        <source>
        
        Inputs:
        <input1> <units>
        <input2> <units>
        ..
        
        Outputs:
        <output1> <units>
        <output2> <units>
        ..
        
        Properties Used:
        <property1> <units>
        <property2> <units>
        ..
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
#  Attribute Container
# ----------------------------------------------------------------------
## @ingroup templates
class Container(Container):
    """<Description>
    
    Assumptions:
    <any assumptions>
    
    Source:
    <source>
    """
    pass

# add to attribute
Example_Attribute.Container = Container        
        
        
# ----------------------------------------------------------------------
#   Helper Functions
# ----------------------------------------------------------------------
# these will not be available in the SUAVE namespace
## @ingroup templates
def helper_function(input1,inputs2=None):
    """<Description>

    Assumptions:
    <any assumptions>

    Source:
    <source>

    Inputs:
    <input1> <units>
    <input2> <units>
    ..

    Outputs:
    <output1> <units>
    <output2> <units>
    ..

    Properties Used:
    N/A
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
    raise RuntimeError('test failed, not implemented')





