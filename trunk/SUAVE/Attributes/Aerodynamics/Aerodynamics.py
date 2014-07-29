# Aerodynamics.py
# 
# Created:  Trent, Nov 2013
# Modified: Trent, Anil, Tarik, Feb 2014       


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data
from Configuration   import Configuration
from Conditions      import Conditions
from Geometry        import Geometry

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np
import scipy as sp

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Aerodynamics(Data):
    """ SUAVE.Attributes.Aerodynamics.Aerodynamics
        base class for a vehicle aerodynamics model
        
        this class is callable
    
    """
    def __defaults__(self):
        self.tag = 'Aerodynamics'
        self.geometry      = Geometry()
        self.configuration = Configuration()
        self.stability     = None
        
    def __call__(self,conditions):
        """ process vehicle to setup geometry, condititon and configuration
            
            Inputs:
                conditions - DataDict() of aerodynamic conditions
                
            Outputs:
                CL - array of lift coefficients, same size as alpha 
                CD - array of drag coefficients, same size as alpha
                
            Assumptions:
                no changes to initial geometry or configuration
                
        """
        
        # calculate aerodynamics
        CL = self.calculate_lift(conditions,self.configuration,self.geometry)
        CD = self.calculate_drag(conditions,self.configuration,self.geometry)
    
        return CL, CD
        
    
    # ------------------------------------------------------------------
    #   Raw Aerodynamics Functions
    # ------------------------------------------------------------------
    
    # these functions are static methods and cannot querey self for data
    # this is to allow users to monkey patch their desired lift function
    
    @staticmethod
    def calculate_lift(conditions,configuration,geometry):
        raise NotImplementedError
        return CL
    
    
    @staticmethod
    def calculate_drag(conditions,configuration,geometry):
        raise NotImplementedError
        return CD
