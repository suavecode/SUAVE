from SUAVE.Core import Data

#from SUAVE.Analyses.Aerodynamics import Aerodynamics
#from SUAVE.Analyses              import Surrogate




#  DON'T FORGET TO ADD THIS TO __INIT__.PY WHEN IT'S WORKING!

class AVL(Data):
    ''' this class only builds and evaluates an avl surrogate of aerodynamics
        it must be patched into a markup analysis if more fidelity is needed
    '''
    def __defaults__(self):

        self.training.angle_of_attack  = np.array([-10.,0.,10.]) * Units.deg
        self.training.lift_coefficient = None
        self.training.drag_coefficient = None
        self.training.pitching_moment_coefficient = None

        self.surrogates = Data()
        self.surrogates.lift_coefficient = None
        self.surrogates.induced_drag_coefficient = None
        self.surrogates.pitching_moment_coefficient = None

        self.avl_callable = AVL_Callable()
        self.avl_callable.keep_files = False
        
        return


    def initialize(self,vehicle):
        
        self.avl_callable.initialize(self.geometry)
        self.sample_training()
        self.build_surrogate()
        
        return


    def sample_training(self):

        # define conditions for run cases
        run_conditions = Aerodynamics()
        ones_1col      = run_conditions.ones_row(1)
        run_conditions.weights.total_mass = ones_1col*vehicle.mass_properties.max_takeoff
        run_conditions.freestream.mach_number = ones_1col * 0.0
        run_conditions.freestream.velocity    = ones_1col * 150 * Units.knots
        run_conditions.freestream.density     = ones_1col * 1.225
        run_conditions.freestream.gravity     = ones_1col * 9.81
        
        # set up run cases
        alphas = self.training.angle_of_attack
        run_conditions.expand_rows(alphas.shape[0])
        run_conditions.aerodynamics.angle_of_attack = alphas

        # run avl
        results = avl_callable(run_conditions)
        self.training.lift_coefficient = results.aerodynamics.total_lift_coefficient
        self.training.induced_drag_coefficient = \
            results.aerodynamics.induced_drag_coefficient
        self.training.pitching_moment_coefficient = \
            results.aerodynamics.pitch_moment_coefficient

        return


    def build_surrogate(self):
        # unpack
        training_data = self.training
        AoA_data = training_data.angle_of_attack
        CL_data  = training_data.lift_coefficient
        CDi_data = training_data.induced_drag_coefficient
        Cm_data  = training_data.pitching_moment_coefficient

        # pack for surrogate
        X_data = np.reshape(AoA_data,-1)

        # assign models
        lift_model  = np.poly1d(np.polyfit(X_data,CL_data,1))
        drag_model  = np.poly1d(np.polyfit(X_data,CDi_data,2))
        pitch_model = np.poly1d(np.polyfit(X_data,Cm_data,1))

        # populate surrogates
        self.surrogates.lift_coefficient = lift_model
        self.surrogates.induced_drag_coefficient = drag_model
        self.surrogates.pitching_moment_coefficient = pitch_model

        return


    def evaluate(self,state):
        # unpack
        aoa           = state.conditions.freestream.angle_of_attack
        q             = state.conditions.freestream.dynamic_pressure
        Sref          = self.geometry.reference_area

        # evaluate surrogates
        CL  = self.surrogates.lift_coefficient(aoa)
        CDi = self.surrogates.induced_drag_coefficient(aoa)
        Cm  = self.surrogates.pitching_moment_coefficient(aoa)

        # pack conditions
        state.conditions.aerodynamics.lift_coefficient = CL
        state.conditions.aerodynamics.drag_coefficient = CDi
        state.conditions.aerodyanmics.pitching_moment_coefficient = Cm

        # pack results
        results = Data()
        results.lift_coefficient = CL
        results.induced_drag_coefficient = CDi
        results.pitching_moment_coefficient = Cm

        return results


    def evaluate_lift(self,state):
        # unpack
        aoa   = state.conditions.freestream.angle_of_attack
        q     = state.conditions.freestream.dynamic_pressure
        Sref  = self.geometry.reference_area

        # evaluate surrogates
        CL  = self.surrogates.lift_coefficient(aoa)

        # pack conditions
        state.conditions.aerodynamics.lift_coefficient = CL

        # pack results
        results = Data()
        results.lift_coefficient = CL

        return results