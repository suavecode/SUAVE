# Aerodynamics_Surrogate.py
# 
# Created:  Trent, Nov 2013
# Modified: Trent, Anil, Tarik, Feb 2014       


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Core import Data
from SUAVE.Attributes import Units

# local imports
from Aerodynamics    import Aerodynamics
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

class Aerodynamics_Surrogate(Aerodynamics):
    """ SUAVE.Attributes.Aerodynamics.Aerodynamics_Surrogate
        aerodynamic model that builds a surrogate model
        
        this class is callable, see self.__call__
        
    """
    
    def __defaults__(self):
        
        self.tag = 'Aerodynamics'
        self.geometry      = Geometry()
        self.configuration = Configuration()
        
        self.conditions_table = Conditions(
            angle_of_attack = np.linspace(-10., 10.  , 5) * Units.deg ,
            mach_number     = np.linspace(0.0 , 1.0  , 5) ,
            reynolds_number = np.linspace(1.e4, 1.e10, 3) ,
        )
        
        self.model = Data()
        
        
    def initialize(self):
        
        # build the conditions table
        self.build_conditions_table()
        
        # unpack
        conditions_table = self.conditions_table
        geometry         = self.geometry
        configuration    = self.configuration
        #
        AoA = conditions_table.angle_of_attack
        Ma  = conditions_table.mach_number
        Re  = conditions_table.reynolds_number
        
        # check
        assert len(AoA) == len(Ma) == len(Re) , 'condition length mismatch'
        
        n_conditions = len(AoA)
        
        # arrays
        CL  = np.zeros_like(AoA)
        CD  = np.zeros_like(AoA)
        
        # condition input, local, do not keep
        konditions = Conditions()
        
        # calculate aerodynamics for table
        for i in xrange(n_conditions):
            
            # overriding conditions, thus the name mangling
            konditions.angle_of_attack = AoA[i]
            konditions.mach_number     = Ma[i]
            konditions.reynolds_number = Re[i]
            
            # these functions are inherited from Aerodynamics() or overridden
            CL[i] = self.calculate_lift(konditions, configuration, geometry)
            CD[i] = self.calculate_drag(konditions, configuration, geometry)
            
        # store table
        conditions_table.lift_coefficient = CL
        conditions_table.drag_coefficient = CD
        
        # build surrogate
        self.build_surrogate()
        
        return
        
    #: def initialize()
    
    def build_conditions_table(self):
        
        conditions_table = self.conditions_table
        
        # unpack, check unique just in case
        AoA_list = np.unique( conditions_table.angle_of_attack )
        Ma_list  = np.unique( conditions_table.mach_number     )
        Re_list  = np.unique( conditions_table.reynolds_number )
        
        # mesh grid, beware of dimensionality!
        AoA_list,Ma_list,Re_list = np.meshgrid(AoA_list,Ma_list,Re_list)
        
        # need 1D lists
        AoA_list = np.reshape(AoA_list,-1)
        Ma_list  = np.reshape(Ma_list,-1)
        Re_list  = np.reshape(Re_list,-1)
        
        # repack conditions table
        conditions_table.angle_of_attack = AoA_list
        conditions_table.mach_number     = Ma_list
        conditions_table.reynolds_number = Re_list
        
        return
    
    #: def build_conditions_table()    
    
    def build_surrogate(self):
        
        # unpack data
        conditions_table = self.conditions_table
        AoA_data = conditions_table.angle_of_attack
        Ma_data  = conditions_table.mach_number
        Re_data  = conditions_table.reynolds_number
        #
        CL_data  = conditions_table.lift_coefficient
        CD_data  = conditions_table.drag_coefficient
        
        # reynolds log10 space!
        Re_data = np.log10(Re_data)
        
        # pack for surrogate
        X_data = np.array([AoA_data,Ma_data,Re_data]).T        
        
        # assign models
        Interpolation = Aerodynamics_Surrogate.Interpolation
        self.model.lift_coefficient = Interpolation(X_data,CL_data)
        self.model.drag_coefficient = Interpolation(X_data,CD_data)
        
        return
        
    #: def build_surrogate()
        
    def __call__(self,conditions):
        """ process vehicle to setup geometry, condititon and configuration
            
            Inputs:
                conditions - DataDict() of aerodynamic conditions
                
            Outputs:
                CL - array of lift coefficients, same size as alpha 
                CD - array of drag coefficients, same size as alpha
                
            Assumptions:
                linear intperolation surrogate model on Mach, Angle of Attack 
                    and Reynolds number
                locations outside the surrogate's table are held to nearest data
                no changes to initial geometry or configuration
                
        """
        
        # unpack conditions
        AoA = np.atleast_1d( conditions.angle_of_attack )
        Ma  = np.atleast_1d( conditions.mach_number )
        Re  = np.atleast_1d( conditions.reynolds_number )
        
        # reynolds log10 space!
        Re = np.log10(Re)
        
        # pack for interpolate
        X_interp = np.array([AoA,Ma,Re]).T        
        
        # interpolate
        CL = self.model.lift_coefficient(X_interp)
        CD = self.model.drag_coefficient(X_interp)        
        
        return CL, CD
        
    #: def __call__()
    
#: class Aerodynamics_Surrogate()


# ----------------------------------------------------------------------
#  Helper Class - Interpolation
# ----------------------------------------------------------------------

class Interpolation(object):
    
    def __init__(self,X_data,C_data):
        
        # build interpolators
        self.linear_interp  = sp.interpolate.LinearNDInterpolator(X_data,C_data)
        self.nearest_interp = sp.interpolate.NearestNDInterpolator(X_data,C_data)
        #self.rbf_interp     = sp.interpolate.Rbf(AoA_data,Ma_data,Re_data,C_data,
                                                 #function='gaussian',smooth=True)
        
    def __call__(self,X_interp):
        
        # interpolate linear
        C_interp = self.linear_interp(X_interp)
        
        # safe gaurd for interpolation outside of data support
        i_nan = np.isnan(C_interp)
        C_interp[i_nan] = self.nearest_interp(X_interp[i_nan]) # TODO: smooth infeasibility  
        
         #warning
        if np.any(i_nan): warn('interpolation outside training data',Warning)        
        
        #C_interp = self.rbf_interp(AoA,Ma,Re)
        
        return C_interp
    
    ## radial basis function surrogate
    #model = self.model
    #model.lift_coefficent  = sp.interpolate.Rbf(AoA_data,Ma_data,Re_data,CL_data,
                                                #function=self.model_type)
    #model.drag_coefficient = sp.interpolate.Rbf(AoA_data,Ma_data,Re_data,CD_data,
                                                #function=self.model_type)
Aerodynamics_Surrogate.Interpolation = Interpolation

# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__': 
    print 'Test %s' % __file__
    
    aerodynamics = Aerodynamics_Surrogate()
    
    def calculate_drag(conditions,configuration=None,geometry=None):
        return ( conditions.angle_of_attack * 2.*np.pi ) ** 2.
    
    def calculate_lift(conditions,configuration=None,geometry=None):
        return conditions.angle_of_attack * 2.*np.pi
    
    # monkey patch the models
    aerodynamics.calculate_drag = calculate_drag
    aerodynamics.calculate_lift = calculate_lift
    
    aerodynamics.initialize()
    
    # test conditions
    conditions = Conditions(
        angle_of_attack = 8.0 * Units.deg ,
        mach_number     = 0.5  ,
        reynolds_number = 1.e6,
    )
    
    # call the surrogate
    CL_I,CD_I = aerodynamics.__call__(conditions)
    
    # truth data
    CL_T = aerodynamics.calculate_lift(conditions)
    CD_T = aerodynamics.calculate_drag(conditions)
    
    print 'CL = %.4f = %.4f' % (CL_I[0],CL_T)
    print 'CD = %.4f = %.4f' % (CD_I[0],CD_T)
    
