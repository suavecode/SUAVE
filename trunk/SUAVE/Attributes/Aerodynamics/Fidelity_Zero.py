# Fidelity_Zero.py
# 
# Created:  Trent, Nov 2013
# Modified: Trent, Anil, Tarik, Feb 2014       


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Structure import Data
from SUAVE.Attributes import Units

from SUAVE.Methods.Aerodynamics.Lift.weissenger_vortex_lattice import weissinger_vortex_lattice
from SUAVE.Methods.Aerodynamics.Lift import compute_aircraft_lift
from SUAVE.Methods.Aerodynamics.Drag import compute_aircraft_drag

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

class Fidelity_Zero(Aerodynamics_Surrogate):
    """ SUAVE.Attributes.Aerodynamics.Fidelity_Zero
        aerodynamic model that builds a surrogate model for clean wing 
        lift, using vortex lattic, and various handbook methods
        for everything else
        
        this class is callable, see self.__call__
        
    """
    
    def __defaults__(self):
        
        self.tag = 'Fidelity_Zero'
        
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
            angle_of_attack = np.linspace(-10., 10., 5) * Units.deg ,
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
        geometry.reference_area = vehicle.Sref
        
        # arrays
        CL  = np.zeros_like(AoA)
        
        # condition input, local, do not keep
        konditions = Conditions()
        
        # calculate aerodynamics for table
        for i in xrange(n_conditions):
            
            # overriding conditions, thus the name mangling
            konditions.angle_of_attack = AoA[i]
            
            # these functions are inherited from Aerodynamics() or overridden
            CL[i] = calculate_lift_vortex_lattice(konditions, configuration, geometry)
            
        # store table
        conditions_table.lift_coefficient = CL
        
        # build surrogate
        self.build_surrogate()
        
        return
        
    #: def initialize()

    # don't need to build a conditions table
    build_conditions_table = None
    
    
    def build_surrogate(self):
        
        # unpack data
        conditions_table = self.conditions_table
        AoA_data = conditions_table.angle_of_attack
        #
        CL_data  = conditions_table.lift_coefficient
        
        # pack for surrogate
        X_data = np.array([AoA_data]).T        
        
        # assign models
        Interpolation = Fidelity_Zero.Interpolation
        self.models.lift_coefficient = Interpolation(X_data,CL_data)
        
        # assign to configuration
        self.configuration.surrogate_models = self.models
                
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
        
        #Mc  = conditions.mach_number
        #roc = conditions.density
        #muc = conditions.viscosity
        #Tc  = conditions.temperature    
        #pc  = conditions.pressure
        #aoa = conditions.angle_of_attack        
        
        # unpack
        configuration = self.configuration
        geometry      = self.geometry
                
        # lift needs to compute first, updates data needed for drag
        CL = compute_aircraft_lift(conditions,configuration,geometry)
        
        # drag computes second
        CD = compute_aircraft_drag(conditions,configuration,geometry)
        
        return CL, CD
        
    #: def __call__()
    
def calculate_lift_vortex_lattice(conditions,configuration,geometry):
    """ calculate total vehicle lift coefficient by vortex lattice
    """
    
    # unpack
    vehicle_reference_area = geometry.Sref
    
    total_lift = 0.0
    
    for wing in geometry.Wings:
        wing_lift = weissinger_vortex_lattice(conditions,configuration,wing)
        total_lift += wing_lift * wing.Sref / vehicle_reference_area
    
    return total_lift
    
    
    
#: class Aerodynamics_Surrogate()

