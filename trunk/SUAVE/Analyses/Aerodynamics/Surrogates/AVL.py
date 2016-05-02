# AVL_Surrogate.py
#
# Created:  Tim Momose, March 2015 
# Modified: Feb 2016, Andrew Wendorff


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

from SUAVE.Core import Data, Units
from SUAVE.Analyses.Aerodynamics.Aerodynamics import Aerodynamics
from SUAVE.Analyses.Aerodynamics import AVL as AVL_Callable
from SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics\
     import Aerodynamics as Aero_Conditions

from SUAVE.Analyses import Surrogate

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

class AVL(Aerodynamics,Surrogate):
    ''' This class only builds and evaluates an avl surrogate of aerodynamics
        It must be patched into a markup analysis if more fidelity is needed.
        The surrogate models lift coefficient, induced drag coefficient, and
        pitching moment coefficient versus angle of attack.
    '''
    def __defaults__(self):
        
        self.training = Data()
        self.training.angle_of_attack  = np.array([-10.,0.,10.]) * Units.deg
        self.training.lift_coefficient = None
        self.training.drag_coefficient = None
        self.training.pitch_moment_coefficient = None

        self.surrogates = Data()
        self.surrogates.lift_coefficient = None
        self.surrogates.induced_drag_coefficient = None
        self.surrogates.pitch_moment_coefficient = None

        self.avl_callable = AVL_Callable()
        self.avl_callable.keep_files = False
        self.avl_callable.settings.filenames.run_folder = 'avl_surrogate_files'
        
        self.geometry = None
        
        self.finalized = False
        
        return


    def finalize(self):
        
        if not self.finalized:
            
            print 'Building AVL Surrogate'
            
            self.avl_callable.features = self.geometry
            self.avl_callable.finalize()
            self.sample_training()
            self.build_surrogate()
        
            self.finalized = True
            
            print 'Done'
        
        return

    initialize = finalize
    
    def sample_training(self):

        # define conditions for run cases
        run_conditions = Aero_Conditions()
        ones_1col      = run_conditions.ones_row(1)
        run_conditions.weights.total_mass     = ones_1col*self.geometry.mass_properties.max_takeoff
        run_conditions.freestream.mach_number = ones_1col * 0.0
        run_conditions.freestream.velocity    = ones_1col * 150 * Units.knots
        run_conditions.freestream.density     = ones_1col * 1.225
        run_conditions.freestream.gravity     = ones_1col * 9.81
        
        # set up run cases
        alphas_1d = self.training.angle_of_attack
        alphas    = alphas_1d.reshape([alphas_1d.shape[0],1])
        run_conditions.expand_rows(alphas.shape[0])
        run_conditions.aerodynamics.angle_of_attack = alphas

        # run avl
        results = self.avl_callable.evaluate_conditions(run_conditions)
        self.training.lift_coefficient = results.aerodynamics.lift_coefficient.reshape(alphas_1d.shape)
        self.training.induced_drag_coefficient = \
            results.aerodynamics.drag_breakdown.induced.total.reshape(alphas_1d.shape)
        self.training.pitch_moment_coefficient = \
            results.aerodynamics.pitch_moment_coefficient.reshape(alphas_1d.shape)

        return


    def build_surrogate(self):
        
        # unpack
        training_data = self.training
        AoA_data = training_data.angle_of_attack
        CL_data  = training_data.lift_coefficient
        CDi_data = training_data.induced_drag_coefficient
        Cm_data  = training_data.pitch_moment_coefficient

        # pack for surrogate
        X_data = np.reshape(AoA_data,-1)

        # assign models
        lift_model  = np.poly1d(np.polyfit(X_data,CL_data,1))
        drag_model  = np.poly1d(np.polyfit(X_data,CDi_data,2))
        pitch_model = np.poly1d(np.polyfit(X_data,Cm_data,1))

        # populate surrogates
        self.surrogates.lift_coefficient = lift_model
        self.surrogates.induced_drag_coefficient = drag_model
        self.surrogates.pitch_moment_coefficient = pitch_model

        return


    def evaluate(self,state,settings=None,geometry=None):
        
        # unpack
        aoa           = state.conditions.aerodynamics.angle_of_attack
        Sref          = self.geometry.reference_area

        # evaluate surrogates
        CL  = self.surrogates.lift_coefficient(aoa)
        CDi = self.surrogates.induced_drag_coefficient(aoa)
        Cm  = self.surrogates.pitch_moment_coefficient(aoa)

        # pack conditions
        state.conditions.aerodynamics.lift_coefficient = CL
        state.conditions.aerodynamics.drag_coefficient = CDi
        state.conditions.aerodynamics.pitch_moment_coefficient = Cm

        # pack results
        results = Data()
        results.lift_coefficient = CL
        results.drag_coefficient = CDi
        results.induced_drag_coefficient = CDi
        results.pitch_moment_coefficient = Cm
        
        #results.update( self.compute_forces(state.conditions) )

        return results


    def evaluate_lift(self,state):
        
        # unpack
        aoa   = state.conditions.aerodynamics.freestream.angle_of_attack
        Sref  = self.geometry.reference_area

        # evaluate surrogates
        CL  = self.surrogates.lift_coefficient(aoa)

        # pack conditions
        state.conditions.aerodynamics.lift_coefficient = CL

        # pack results
        results = Data()
        results.lift_coefficient = CL

        return results