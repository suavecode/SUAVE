# Conditions_polar.py
# 
# Created:  Trent, Anil, Tarik, Feb 2014
# Modified:         


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
from SUAVE.Core import \
    Data, Container, Data_Exception, Data_Warning
from SUAVE.Attributes.Results import Result

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np
import scipy as sp

#Data.update(self)

# ----------------------------------------------------------------------
#  Attribute
# ----------------------------------------------------------------------

class Conditions_polar(Data):
    """ SUAVE.Attributes.Aerodynamics.Conditions()
        conditions data for a aerodynamics evaluation
        
        Attributes:
            none
            
        Methods:
            none
            
        Assumptions:
            unless explicitly named otherwise, all values are for 
                total vehicle or freestream
        
    """
    
    def __defaults__(self):
        #self.angle_of_attack = []
        #self.mach_number = []
        
        #self.lift_breakdown = Result.Container(
            #total=[] 
        #)
        #self.drag_breakdown = Result.Container(
            #total=[] 
        #
        self.freestream = Data()
        self.aerodynamics = Data()
        self.freestream.mach_number = 0.0
        self.aerodynamics.angle_of_attack = 0.0
        
        pass
        
        
    def __check__(self):
        pass
    




