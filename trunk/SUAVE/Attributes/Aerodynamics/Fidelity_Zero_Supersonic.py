# Fidelity_Zero.py
# 
# Created:  Trent, Nov 2013
# Modified: Trent, Anil, Tarik, Feb 2014       


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports

import SUAVE

from SUAVE.Structure import Data
from SUAVE.Attributes import Units

from SUAVE.Methods.Aerodynamics.Lift.weissenger_vortex_lattice import weissinger_vortex_lattice
from SUAVE.Methods.Aerodynamics.Lift.linear_supersonic_lift import linear_supersonic_lift
from SUAVE.Methods.Aerodynamics.Drag import compute_aircraft_drag_supersonic
from SUAVE.Attributes.Aerodynamics.Aerodynamics_1d_Surrogate import Aerodynamics_1d_Surrogate


# local imports
from Aerodynamics_Surrogate import Aerodynamics_Surrogate
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

class Fidelity_Zero_Supersonic(Aerodynamics_Surrogate):
    """ SUAVE.Attributes.Aerodynamics.Fidelity_Zero
        aerodynamic model that builds a surrogate model for clean wing 
        lift, using vortex lattic, and various handbook methods
        for everything else
        
        this class is callable, see self.__call__
        
    """
    
    def __defaults__(self):
        
        self.tag = 'Fidelity_Zero_Supersonic'
        
        self.geometry      = Geometry()
        
        self.configuration = Configuration()

        # correction factors
        self.configuration.fuselage_lift_correction           = 1.14
        self.configuration.trim_drag_correction_factor        = 1.02
        self.configuration.wing_parasite_drag_form_factor     = 1.1
        self.configuration.fuselage_parasite_drag_form_factor = 2.3
        self.configuration.aircraft_span_efficiency_factor    = 0.78
        
        # vortex lattice configurations
        self.configuration.number_panels_spanwise  = 5
        self.configuration.number_panels_chordwise = 1
        
        self.conditions_table = Conditions(
            angle_of_attack = np.array([-10,-5,0,5,10.0]) * Units.deg ,
        )
        
        self.models = Data()
        
        
    def initialize(self,vehicle):
                        
        # unpack
        conditions_table = self.conditions_table
        geometry         = self.geometry
        configuration    = self.configuration
        #
        AoA = conditions_table.angle_of_attack
        n_conditions = len(AoA)
        
        # copy geometry
        for k in ['Fuselages','Wings','Propulsors']:
            geometry[k] = deepcopy(vehicle[k])
        
        # reference area
        geometry.reference_area = vehicle.S
              
        
        # arrays
        CL  = np.zeros_like(AoA)
        
        # condition input, local, do not keep
        konditions = Conditions()
        konditions.aerodynamics = Data()
        
        # calculate aerodynamics for table
        for i in xrange(n_conditions):
            
            # overriding conditions, thus the name mangling
            konditions.aerodynamics.angle_of_attack = AoA[i]
            
            # these functions are inherited from Aerodynamics() or overridden
            CL[i] = calculate_lift_vortex_lattice(konditions, configuration, geometry)
            
        # store table
        conditions_table.lift_coefficient = CL
        
        # build surrogate
        self.build_surrogate_sub()
        
        # calculate aerodynamics for table
        for i in xrange(n_conditions):
            
            # overriding conditions, thus the name mangling
            konditions.aerodynamics.angle_of_attack = AoA[i]
            
            # these functions are inherited from Aerodynamics() or overridden
            CL[i] = calculate_lift_linear_supersonic(konditions, configuration, geometry)
            
        # store table
        conditions_table.lift_coefficient = CL
        
        # build surrogate
        self.build_surrogate_sup()        
        
        return
        
    #: def initialize()    
    def build_surrogate_sub(self):
        
        # unpack data
        conditions_table = self.conditions_table
        AoA_data = conditions_table.angle_of_attack
        #
        CL_data  = conditions_table.lift_coefficient
        
        # pack for surrogate
        X_data = np.array([AoA_data]).T        
        
        # assign models
        X_data = np.reshape(X_data,-1)
        # assign models
        #Interpolation = Aerodynamics_1d_Surrogate.Interpolation(X_data,CL_data)
        Interpolation = np.poly1d(np.polyfit(X_data, CL_data ,1))
        
        #Interpolation = Fidelity_Zero.Interpolation
        self.models.lift_coefficient = Interpolation
        
        # assign to configuration
        self.configuration.surrogate_models_sub = Data()
        self.configuration.surrogate_models_sub.lift_coefficient = Interpolation
                
        return
    
    def build_surrogate_sup(self):
        
        # unpack data
        conditions_table = self.conditions_table
        AoA_data = conditions_table.angle_of_attack
        #
        CL_data  = conditions_table.lift_coefficient
        
        # pack for surrogate
        X_data = np.array([AoA_data]).T        
        
        # assign models
        X_data = np.reshape(X_data,-1)
        # assign models
        #Interpolation = Aerodynamics_1d_Surrogate.Interpolation(X_data,CL_data)
        Interpolation = np.poly1d(np.polyfit(X_data, CL_data ,1))
        
        #Interpolation = Fidelity_Zero.Interpolation
        self.models.lift_coefficient = Interpolation
        
        # assign to configuration
        self.configuration.surrogate_models_sup = Data()
        self.configuration.surrogate_models_sup.lift_coefficient = Interpolation
                
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
        
        # unpack
        configuration = self.configuration
        geometry      = self.geometry
        q             = conditions.freestream.dynamic_pressure
        Sref          = geometry.reference_area
                
        # lift needs to compute first, updates data needed for drag
        CL = SUAVE.Methods.Aerodynamics.Lift.compute_aircraft_lift_supersonic(conditions,configuration,geometry)
        

        
        # drag computes second
        CD = compute_aircraft_drag_supersonic(conditions,configuration,geometry)
        
        
        # pack conditions
        conditions.aerodynamics.lift_coefficient = CL
        conditions.aerodynamics.drag_coefficient = CD        

        # pack results
        results = Data()
        results.lift_coefficient = CL
        results.drag_coefficient = CD
        
        N = q.shape[0]
        L = np.zeros([N,3])
        D = np.zeros([N,3])
        
        L[:,2] = ( -CL * q * Sref )[:,0]
        D[:,0] = ( -CD * q * Sref )[:,0]
        
        #if L.any < 0.0:
            #k = 1        

        results.lift_force_vector = L
        results.drag_force_vector = D        
        
        return results
        
    #: def __call__()
    
    # don't need to build a conditions table
    build_conditions_table = None
    
    
    
def calculate_lift_vortex_lattice(conditions,configuration,geometry):
    """ calculate total vehicle lift coefficient by vortex lattice
    """
    
    # unpack
    vehicle_reference_area = geometry.reference_area
    
    # iterate over wings
    total_lift_coeff = 0.0
    for wing in geometry.Wings.values():
        
        [wing_lift_coeff,wing_drag_coeff] = weissinger_vortex_lattice(conditions,configuration,wing)
        total_lift_coeff += wing_lift_coeff * wing.Areas.reference / vehicle_reference_area

    return total_lift_coeff
    
def calculate_lift_linear_supersonic(conditions,configuration,geometry):
    """ Calculate total vehicle lift coefficient using linear supersonic theory
    """
    
    # unpack
    vehicle_reference_area = geometry.reference_area
    
    # iterate over wings
    total_lift_coeff = 0.0
    for wing in geometry.Wings.values():
        
        wing_lift_coeff = linear_supersonic_lift(conditions,configuration,wing)
        total_lift_coeff += wing_lift_coeff * wing.Areas.reference / vehicle_reference_area
    
    return total_lift_coeff    



