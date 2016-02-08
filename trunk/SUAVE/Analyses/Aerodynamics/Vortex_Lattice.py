# Vortex_Lattice.py
#
# Created:  Trent, Nov 2013
# Modified: Trent, Anil, Tarik, Feb 2014
# Modified: Trent, Jan 2014  
# Modified: Feb 2016, Andrew Wendorff


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
import SUAVE

from SUAVE.Core import Data
from SUAVE.Core import Units

from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift import weissinger_vortex_lattice


# local imports
from Aerodynamics import Aerodynamics


# package imports
import numpy as np


# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Vortex_Lattice(Aerodynamics):
    """ SUAVE.Analyses.Aerodynamics.Fidelity_Zero
        aerodynamic model that builds a surrogate model for clean wing
        lift, using vortex lattice, and various handbook methods
        for everything else

        this class is callable, see self.__call__

    """

    def __defaults__(self):

        self.tag = 'Vortex_Lattice'

        self.geometry = Data()
        self.settings = Data()

        # correction factors
        self.settings.fuselage_lift_correction           = 1.14
        self.settings.trim_drag_correction_factor        = 1.02
        self.settings.wing_parasite_drag_form_factor     = 1.1
        self.settings.fuselage_parasite_drag_form_factor = 2.3
        self.settings.aircraft_span_efficiency_factor    = 0.78
        self.settings.drag_coefficient_increment         = 0.0000

        # vortex lattice configurations
        self.settings.number_panels_spanwise  = 5
        self.settings.number_panels_chordwise = 1

        # conditions table, used for surrogate model training
        self.training = Data()        
        self.training.angle_of_attack  = np.array([-10.,-5.,0.,5.,10.]) * Units.deg
        self.training.lift_coefficient = None
        
        # surrogoate models
        self.surrogates = Data()
        self.surrogates.lift_coefficient = None
 
        
    def initialize(self):
                   
        # sample training data
        self.sample_training()
                    
        # build surrogate
        self.build_surrogate()


    def evaluate(self,state,settings,geometry):
        """ process vehicle to setup geometry, condititon and settings

            Inputs:
                conditions - DataDict() of aerodynamic conditions

            Outputs:
                CL - array of lift coefficients, same size as alpha
                CD - array of drag coefficients, same size as alpha

            Assumptions:
                linear intperolation surrogate model on Mach, Angle of Attack
                    and Reynolds number
                locations outside the surrogate's table are held to nearest data
                no changes to initial geometry or settings
        """

        # unpack

        surrogates = self.surrogates        

        conditions = state.conditions
        
        # unpack

        
        q    = conditions.freestream.dynamic_pressure
        AoA  = conditions.aerodynamics.angle_of_attack
        Sref = geometry.reference_area
        
        wings_lift_model = surrogates.lift_coefficient
        
        # inviscid lift of wings only
        inviscid_wings_lift                                        = wings_lift_model(AoA)
        conditions.aerodynamics.lift_breakdown.inviscid_wings_lift = inviscid_wings_lift
        state.conditions.aerodynamics.lift_coefficient             = inviscid_wings_lift


        return inviscid_wings_lift


    def sample_training(self):
        
        # unpack
        geometry = self.geometry
        settings = self.settings
        training = self.training
        
        AoA = training.angle_of_attack
        CL  = np.zeros_like(AoA)

        # condition input, local, do not keep
        konditions              = Data()
        konditions.aerodynamics = Data()

        # calculate aerodynamics for table
        for i,_ in enumerate(AoA):
            
            # overriding conditions, thus the name mangling
            konditions.aerodynamics.angle_of_attack = AoA[i]
            
            # these functions are inherited from Aerodynamics() or overridden
            CL[i] = calculate_lift_vortex_lattice(konditions, settings, geometry)

        # store training data
        training.lift_coefficient = CL

        return

    def build_surrogate(self):

        # unpack data
        training = self.training
        AoA_data = training.angle_of_attack
        CL_data  = training.lift_coefficient

        # pack for surrogate model
        X_data = np.array([AoA_data]).T
        X_data = np.reshape(X_data,-1)
        
        # learn the model
        cl_surrogate = np.poly1d(np.polyfit(X_data, CL_data ,1))

        #Interpolation = Fidelity_Zero.Interpolation
        self.surrogates.lift_coefficient = cl_surrogate

        return



# ----------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------


def calculate_lift_vortex_lattice(conditions,settings,geometry):
    """ calculate total vehicle lift coefficient by vortex lattice
    """

    # unpack
    vehicle_reference_area = geometry.reference_area

    # iterate over wings
    total_lift_coeff = 0.0
    for wing in geometry.wings.values():

        [wing_lift_coeff,wing_drag_coeff] = weissinger_vortex_lattice(conditions,settings,wing)
        total_lift_coeff += wing_lift_coeff * wing.areas.reference / vehicle_reference_area

    return total_lift_coeff
