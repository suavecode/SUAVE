
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# python imports
import numpy as np
from copy import deepcopy

# SUAVE imports
from SUAVE.Structure                    import Data, Data_Exception
from SUAVE.Structure                    import Container as ContainerBase
from SUAVE.Methods.Utilities.Chebyshev  import chebyshev_data
from SUAVE.Methods.Utilities            import atleast_2d_col

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

class Base_Segment(Data):
    
    # ------------------------------------------------------------------
    #   Data Defaults
    # ------------------------------------------------------------------    

    def __defaults__(self):
        self.tag = 'Base_Segment'
        
        # --- Segment Inputs
        
        # these are the inputs the user will define in the input script
        
        # an example
        ##self.mach_number = 0.7
        #self.mass_initial     = 'previous_segment' ??
        #self.time_initial     = 'previous_segment'
        #self.position_initial = 'previous_segment'
        
        # --- Vehicle Configuration
        
        # a linked copy of the vehicle
        
        self.config = Data()
        
        
        # --- Conditions and Unknowns
        
        # user shouldn't change these in an input script
        # only used for processing / post processing
        # they will be shared with analysis modules and meaningful naming is important
        
        # base array column lengths
        # use a trivial operation to copy the array
        ones_1col = np.ones([1,1])
        ones_2col = np.ones([1,2])
        ones_3col = np.ones([1,3])         
        
        
        # --- Conditions 
        
        # setup conditions
        conditions = Data()
        conditions.frames   = Data()
        conditions.weights  = Data()
        conditions.energies = Data()   
        self.conditions = conditions
        
        # inertial conditions
        conditions.frames.inertial = Data()        
        conditions.frames.inertial.position_vector      = ones_3col * 0
        conditions.frames.inertial.velocity_vector      = ones_3col * 0
        conditions.frames.inertial.acceleration_vector  = ones_3col * 0
        conditions.frames.inertial.total_force_vector   = ones_3col * 0
        conditions.frames.inertial.time                 = ones_1col * 0
        
        # body conditions
        conditions.frames.body = Data()        
        conditions.frames.body.inertial_rotations       = ones_3col * 0
        conditions.frames.body.transform_to_inertial    = np.empty([0,0,0])
        
        # weights conditions
        conditions.weights.total_mass            = ones_1col * 0
        conditions.weights.weight_breakdown      = Data()
        
        # energy conditions
        conditions.energies.total_energy         = ones_1col * 0
        conditions.energies.total_efficiency     = ones_1col * 0
        
        # --- Unknowns
        
        # setup unknowns
        unknowns = Data()
        unknowns.states   = Data()
        unknowns.controls = Data()
        unknowns.finals   = Data()
        self.unknowns = unknowns
        
        # an example
        ## unknowns.states.gamma = ones_1col + 0

        
        # --- Residuals
         
        # setup unknowns
        residuals = Data()
        residuals.states   = Data()
        residuals.controls = Data()
        residuals.finals   = Data()
        self.residuals = residuals
        
        
        # --- Initial Conditions
        
        # this data structure will hold a copy of the last
        #    rows from the conditions of the last segment
        
        self.initials = Data()
        
        
        # --- Numerics
        
        self.numerics = Data()
        self.numerics.tag = 'Solution Numerical Setup'
        
        # discretization
        self.numerics.n_control_points                 = 16
        self.numerics.discretization_method            = chebyshev_data
        
        # solver options
        self.numerics.solver_jacobian                  = "none"
        self.numerics.tolerance_solution               = 1e-8
        self.numerics.tolerance_boundary_conditions    = 1e-8
        
        # dimensionless differentials
        self.numerics.dimensionless_time               = np.empty([0,0])
        self.numerics.differentiate_dimensionless      = np.empty([0,0])
        self.numerics.integrate_dimensionless          = np.empty([0,0])
        
        # time-dimensional differentials
        self.numerics.time                             = np.empty([0,0])
        self.numerics.differentiate_time               = np.empty([0,0])
        self.numerics.integrate_time                   = np.empty([0,0])
        
        return

    # ------------------------------------------------------------------
    #   Methods For Initialization
    # ------------------------------------------------------------------  
    
    def check_inputs(self):
        """ Segment.check():
            error checking of segment inputs
        """
        
        ## CODE
        
        return
    
    def initialize_conditions(self,conditions,numerics,initials=None):
        """ Segment.initialize_conditions(conditions,numerics,initials=None)
            update the segment conditions
            pin down as many condition variables as possible in this function
            
            Inputs:
                conditions - the conditions data dictionary, with initialized zero arrays, 
                             with number of rows = segment.conditions.n_control_points
                initials - a data dictionary with 1-row column arrays pulled from the last row
                           of the previous segment's conditions data, or none if no previous segment
                
            Outputs:
                conditions - the conditions data dictionary, updated with the 
                             values that can be precalculated
            
            Assumptions:
                time_initial comes from either initials.frames.inertial.time[0,0] 
                                            or is set to 0.0
                weight_initial comes from either initialse.weights.total_mass[0,0]
                                              or self.config.Mass_Props.m_takeoff
                
        """
        
        # Usage Notes:
        #     may need to inspect segment (self) for user inputs
        #     arrays will be expanded to number of control points in numerics.n_control_points
        #     will be called before solving the segments free unknowns
        
        
        # process initials
        if initials:
            t_initial = initials.frames.inertial.time[0,0]
            r_initial = initials.frames.inertial.position_vector[0,:][None,:]
            m_initial = initials.weights.total_mass[0,0]
        else:
            t_initial = 0.0
            r_initial = conditions.frames.inertial.position_vector[0,:][None,:]
            m_initial = self.config.Mass_Props.m_takeoff
            
            
        # apply initials
        conditions.weights.total_mass[:,0]   = m_initial
        conditions.frames.inertial.time[:,0] = t_initial
        conditions.frames.inertial.position_vector[:,:] = r_initial[:,:]
        
        return conditions
    
    # ------------------------------------------------------------------
    #   Methods For Solver Iterations
    # ------------------------------------------------------------------    
    
    def update_conditions(self,conditions,numerics,unknowns):
        """ Segment.update_conditions(conditions,numerics,unknowns)
            if needed, updates the conditions given the current free unknowns and numerics
            called once per segment solver iteration
            
            Inputs - 
                unknowns      - data dictionary of segment free unknowns with fields:
                    states, controls, finals
                    these are defined in segment.__defaults__
                conditions    - data dictionary of segment conditions
                    these are defined in segment.__defaults__
                numerics - data dictionary of differential operators for this iteration
                
            Outputs - 
                conditions - data dictionary of updated conditions
                
            Assumptions - 
                preserves the shapes of arrays in conditions

        """
        
        # unpack inputs
        ## CODE
        
        # setup
        ## CODE
        
        # process
        ## CODE
        
        # pack outputs
        ## CODE
        
        return conditions
    
    def solve_residuals(self,conditions,numerics,unknowns,residuals):
        """ Segment.solve_residuals(conditions,numerics,unknowns,residuals)
            the hard work, solves the residuals for the free unknowns
            called once per segment solver iteration
            
            Inputs - 
                unknowns      - data dictionary of segment free unknowns with fields:
                    states, controls, finals
                    these are defined in segment.__defaults__
                conditions    - data dictionary of segment conditions
                    these are defined in segment.__defaults__
                numerics - data dictionary of differential operators for this iteration
                
            Outputs - 
                residuals - data dictionary of residuals, same dictionary structure as unknowns
            
            Usage Notes - 
                after this method, residuals composed into a final residual vector:
                    R = [ [ d(unknowns.states)/dt - residuals.states   ] ;
                          [                         residuals.controls ] ;
                          [                         residuals.finals   ] ] = [0] ,
                    where the segment solver will find a root of R = [0]

        """
        
        # unpack inputs
        ## CODE
        
        # setup
        ## CODE
        
        # process
        ## CODE
        
        # pack outputs
        ## CODE
        
        return residuals
    
    # ------------------------------------------------------------------
    #   Methods For Post-Solver
    # ------------------------------------------------------------------    
    
    def post_process(self,conditions,numerics,unknowns):
        """ Segment.post_process(conditions,numerics,unknowns)
            post processes the conditions after converging the segment solver
            
            Inputs - 
                unknowns - data dictionary of converged segment free unknowns with fields:
                    states, controls, finals
                    these are defined in segment.__defaults__
                conditions - data dictionary of segment conditions
                    these are defined in segment.__defaults__
                numerics - data dictionary of the converged differential operators
                
            Outputs - 
                conditions - data dictionary with remaining fields filled with post-processed conditions
            
            Usage Notes - 
                use this to store the unknowns and any other interesting in conditions
                    for later plotting
                for clarity and style, be sure to define any new fields in segment.__defaults__
            
        """
        
        # unpack inputs
        ## CODE
        
        # setup
        ## CODE
        
        # process
        ## CODE
        
        # pack outputs
        ## CODE
        
        return conditions
    
    
    # ------------------------------------------------------------------
    #   Internal Methods
    # ------------------------------------------------------------------
    # in general it's unlikely these will need to be overidden
    # except for maybe update_differntials() if you are changing 
    #     control point spacing

    def initialize_arrays(self,conditions,numerics,unknowns,residuals):
        """ Segment.initialize_arrays(conditions,numerics,unknowns,residuals)
            expands the number of rows of all arrays in segment.conditions
            and segment.unknowns to numerics.n_control_points
            
            Inputs - 
                conditions - data dictionary of segment conditions                
                residuals  - data dictionary of segment residuals
                unknowns   - data dicitonary of segment unknowns
                numerics   - data dictionary of segment numerics
                
            Outputs -
                unknowns   - updated unknowns
                conditions - updated conditions
                
            Assumptions - 
                updates in-place and recursively through data dictionaries
                requires all dictionary fields to be of type Data() or np.array()
                
        """

        # base array row length
        N = numerics.n_control_points
        ones = np.ones([N,1])
        if numerics.solver_jacobian == 'complex': ones = ones + 0j
        
        
        # recursively initialize condition and unknown arrays 
        # to have row length of number n_control_points
        
        # the update function
        def update_conditions(D):
            for k,v in D.iteritems():
                # recursion
                if isinstance(v,Data):
                    update_conditions(D[k])
                # need arrays here
                elif np.rank(v) == 2:
                    D[k] = np.dot(ones,D[k]) # depends on the ones array here
                #: if type
            #: for each key,value
        #: def update_conditions()
        
        # do the update!
        update_conditions(conditions)
        update_conditions(unknowns.states)
        update_conditions(unknowns.controls)
        update_conditions(residuals.states)
        update_conditions(residuals.controls)
         
        return unknowns,conditions,residuals
    
    def initialize_differentials(self,numerics):
        """ Segment.initialize_differentials(numerics)
            initialize differentiation and integration operator matricies
            
            Inputs - 
                numerics - Data dictionary with fields:
                    dimensionless_time          - empty 2D array
                    differentiate_dimensionless - empty 2D array
                    integrate_dimensionless     - empty 2D array
                    discretization_method       - the method for calculating the above
                    n_control_points            - number of control points for operators
        
            Outputs:
                numerics - Data dictionary with fields:
                    dimensionless_time - time control points, non-dimensional, in range [0,1], column vector
                    differentiate_dimensionless - differention operation matrix
                    integrate_dimensionless - integration operation matrix
                    
                    
            Assumptions:
                operators are in non-dimensional time, with bounds [0,1]
                will call numerics.discretization_method(n_control_points,**numerics) to get operators
                
        """
        
        # unpack
        N                     = numerics.n_control_points
        discretization_method = numerics.discretization_method
        
        # get operators
        t,D,I = discretization_method(N,**numerics)
        t = atleast_2d_col(t)
        
        # pack
        numerics.dimensionless_time          = t
        numerics.differentiate_dimensionless = D
        numerics.integrate_dimensionless     = I
        
        return numerics
    
    def update_differentials(self,conditions,numerics,unknowns=None):
        """ Segment.update_differentials(conditions, numerics, unknowns=None)
            updates the differential operators t, D and I
            must return in dimensional time, with t[0] = 0
            
            Inputs - 
                conditions - data dictionary of segment conditions
                numerics   - data dictionary of non-dimensional differential operators
                unknowns   - data dictionary of segment free unknowns
                
            Outputs - 
                numerics   - udpated data dictionary with dimensional differentials 
                for Base_Segment:
                    numerics.time
                    numerics.differentiate_time
                    numerics.integrate_time
            
            Assumptions - 
                outputed operators are in dimensional time for the current solver iteration
                Base_Segment() by default will rescale operators based on final time,
                    either found in 
                    segment.unknowns.finals.time (will update segment.conditions.frames.inertial.time)
                        otherwise,
                    segment.conditions.frames.inertial.time[-1] - segment.conditions.frames.inertial.time[0]
                
        """
        
        # unpack
        t = numerics.dimensionless_time
        D = numerics.differentiate_dimensionless
        I = numerics.integrate_dimensionless
        
        # rescale time
        if unknowns and unknowns.finals.has_key('time'):
            # variable final time
            dt = unknowns.finals.time
            t = t * dt
            # update inertial time, keep start time
            t_initial = conditions.frames.inertial.time[0]
            conditions.frames.inertial.time = t_initial + t
        else:
            # stationary time control points
            time = conditions.frames.inertial.time
            dt = time[-1] - time[0]
            t = t * dt
        
        # rescale operators
        D = D / dt
        I = I * dt
        
        # pack
        numerics.time               = t
        numerics.differentiate_time = D
        numerics.integrate_time     = I
        
        return numerics

    
    def get_final_conditions(self):
        
        conditions = self.conditions
        finals = Data()
        
        # the update function
        def pull_conditions(A,B):
            for k,v in A.iteritems():
                # recursion
                if isinstance(v,Data):
                    B[k] = Data()
                    pull_conditions(A[k],B[k])
                # need arrays here
                elif not isinstance(v,np.ndarray):
                    raise ValueError , 'condition "%s" must be type np.array' % k
                # the copy
                else:
                    B[k] = A[k][-1,:][None,:]
                #: if type
            #: for each key,value
        #: def pull_conditions()
        
        # do the update!
        pull_conditions(conditions,finals)
        
        return finals
    
    def set_initial_conditions(self,finals):
        
        initials = self.initials
        
        # the update function
        def set_conditions(A,B):
            for k,v in B.iteritems():
                # recursion
                if isinstance(v,Data):
                    set_conditions(A[k],B[k])
                # need arrays here
                elif not isinstance(v,np.ndarray):
                    raise ValueError , 'condition "%s%" must be type np.array' % k
                # the copy
                else:
                    B[k] = A[k][-1,:][:,None]
                #: if type
            #: for each key,value
        #: def pull_conditions()
        
        # do the update!
        set_conditions(finals,initials)
        
        return initials

# ----------------------------------------------------------------------
#  Handle Linking
# ----------------------------------------------------------------------

class Container(ContainerBase):
    pass

Base_Segment.Container = Container
