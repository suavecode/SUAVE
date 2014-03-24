# Conditions.py
# 
# Created:  Trent, Anil, Tarik, Feb 2014
# Modified:         


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
from SUAVE.Structure import \
    Data, Container, Data_Exception, Data_Warning
from SUAVE.Attributes.Results import Result

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

class Conditions(Data):
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
        self.angle_of_attack = []
        self.mach_number = []
        
        self.lift_breakdown = Result.Container(
            total=[] 
        )
        self.drag_breakdown = Result.Container(
            total=[] 
        )
        
    def __check__(self):
        pass
    
# ----------------------------------------------------------------------
#  Conditions Container
# ----------------------------------------------------------------------
class Container(Container):
    
    @staticmethod
    def from_list(angle_of_attack,mach_number,reynolds_number,**data):
        """ class factory, builds a set of conditions from lists or 
            1D arrays of angle of attack, mach number and reynolds number
            can also provide a dictionary of data to add
        """
        
        conditions = zip(angle_of_attack,mach_number,reynolds_number)
        
        new_container = Container()
        
        for condition in conditions:
            this_condition = Conditions(
                angle_of_attack = condition[0],
                mach_number     = condition[1],
                reynolds_number = condition[2],
            )
            
            new_container.append(this_condition)
        
        return new_container
    
    #: def from_list()

# add to attribute
Conditions.Container = Container   
    
# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':    
    raise RuntimeError , 'module test failed, not implemented'





