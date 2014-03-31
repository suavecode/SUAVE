
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
    #   Methods For Initialization
    # ------------------------------------------------------------------    

    def __defaults__(self):
        self.tag = 'Base_Segment'
        
        # --- Segment Inputs
        
        # these are the inputs the user will define in the input script
        
        # an example
        ##self.mach_number = 0.7
        
        
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
        conditions.engergy  = Data()   
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
        conditions.energies.total_energy         = ones_1col + 0
        conditions.energies.total_efficiency     = ones_1col + 0
        
        # --- Unknowns
        
        # setup unknowns
        unknowns = Data()
        unknowns.states   = Data()
        unknowns.controls = Data()
        unknowns.finals   = Data()
        self.unknowns = unknowns
        
        # an example
        ## unknowns.states.gamma = ones_1col + 0
        
        
        # --- Numerics
        
        # numerical options
        self.options = Data()
        self.options.tag = 'Solution Options'
        self.options.n_control_points              = 16
        self.options.jacobian                      = "complex"
        self.options.tolerance_solution            = 1e-8
        self.options.tolerance_boundary_conditions = 1e-8        
        
        # differentials
        self.differentials = Data()
        self.differentials.t = np.empty([0,0]) 
        self.differentials.D = np.empty([0,0]) 
        self.differentials.I = np.empty([0,0]) 
        self.differentials.method = chebyshev_data
        
        return
    
    def check(self):
        """ Segment.check():
            error checking of segment inputs
        """
        
        ## CODE
        
        return
    
    def initialize_conditions(self,conditions):
        """ Segment.initialize_conditions(conditions)
            update the segment conditions
            pin down as many condition variables as possible in this function
            
            Inputs:
                conditions - the conditions data dictionary, with initialized zero arrays, 
                             with number of rows = segment.conditions.n_control_points
                
            Outputs:
                conditions - the conditions data dictionary, updated with the 
                             values that can be precalculated
            
            Assumptions:
                --
                
            Usage Notes:
                may need to inspect segment (self) for user inputs
                will be called before solving the segments free unknowns
                
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
    #   Methods For Solver Iterations
    # ------------------------------------------------------------------    
    
    def update_conditions(unknowns,conditions,differentials):
        """ Segment.update_conditions(unknowns, conditions, differentials)
            if needed, updates the conditions given the current free unknowns and differentials
            called once per segment solver iteration
            
            Inputs - 
                unknowns      - data dictionary of segment free unknowns with fields:
                    states, controls, finals
                    these are defined in segment.__defaults__
                conditions    - data dictionary of segment conditions
                    these are defined in segment.__defaults__
                differentials - data dictionary of differential operators for this iteration
                
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
        
        return residuals
    
    def solve_residuals(unknowns,conditions,differentials):
        """ Segment.solve_residuals(unknowns, conditions, differentials)
            the hard work, solves the residuals for the free unknowns
            called once per segment solver iteration
            
            Inputs - 
                unknowns      - data dictionary of segment free unknowns with fields:
                    states, controls, finals
                    these are defined in segment.__defaults__
                conditions    - data dictionary of segment conditions
                    these are defined in segment.__defaults__
                differentials - data dictionary of differential operators for this iteration
                
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
    
    def post_process(self,unknowns,conditions,differentials):
        """ Segment.post_process(unknowns, conditions, differentials)
            post processes the conditions after converging the segment solver
            
            Inputs - 
                unknowns - data dictionary of converged segment free unknowns with fields:
                    states, controls, finals
                    these are defined in segment.__defaults__
                conditions - data dictionary of segment conditions
                    these are defined in segment.__defaults__
                differentials - data dictionary of the converged differential operators
                
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
        
        return
    
    
    # ------------------------------------------------------------------
    #   Internal Methods
    # ------------------------------------------------------------------
    # in general it's unlikely these will need to be overidden
    # except for maybe update_differntials() if you are changing 
    #     control point spacing

    def initialize_arrays(self,unknowns,conditions,options):
        """ Segment.initialize_arrays(unknowns,conditions,options)
            expands the number of rows of all arrays in segment.conditions
            and segment.unknowns to options.n_control_points
            
            Inputs - 
                unknowns   - data dicitonary of segment unknowns
                conditions - data dictionary of segment conditions
                options    - data dictionary of segment options
                
            Outputs -
                unknowns   - updated unknowns
                conditions - updated conditions
                
            Assumptions - 
                updates in-place and recursively through data dictionaries
                requires all dictionary fields to be of type Data() or np.array()
                
        """

        # base array row length
        N = options.n_control_points
        ones = np.ones([N,1])
        if options.jacobian == 'complex': ones = ones + 0j
        
        # recursively initialize condition and unknown arrays 
        # to have row length of number n_control_points
        
        # the update function
        def update_conditions(D):
            for k,v in D.iteritems():
                # recursion
                if isinstance(v,Data):
                    update_condition(D[k])
                # need arrays here
                elif not isinstance(v,np.ndarray):
                    raise ValueError , 'condition "%s%" must be type np.array' % k
                # the update
                else:
                    D[k] = np.dot(ones,D[k]) # depends on the ones array here
                #: if type
            #: for each key,value
        #: def update_conditions()
        
        # do the update!
        update_conditions(conditions)
        update_conditions(unknowns)
        # like a boss
         
        return unknowns,conditions
    
    def initialize_differentials(self,differentials,options):
        """ Segment.initialize_differentials(differentials)
            initialize differentiation and integration operator matricies
            
            Inputs - 
                differentials - Data dictionary with empty fields:
                    t, D, I
                options - Data dictionary with fields:
                    n_control_points - number of control points for operators
        
            Outputs:
                differentials - Data dictionary with fields:
                    t - time control points, non-dimensional, in range [0,1]
                    D - differention operation matrix
                    I - integration operation matrix
                    method - the method for calculating the above
                    
            Assumptions:
                operators are in non-dimensional time, with bounds [0,1]
                will call differentials.method(n_control_points,**options) to get operators
                
        """
        
        # unpack
        N                   = options.n_control_points
        differential_method = differentials.method
        
        # get operators
        t,D,I = differential_method(N,**options)
        t = atleast_2d_col(t)
        
        # pack
        differentials.t = t
        differentials.D = D
        differentials.I = I
        
        return differentials
    
    def update_differentials(unknowns,conditions,differentials):
        """ Segment.update_differentials(unknowns, conditions, differentials)
            updates the differential operators t, D and I
            must return in dimensional time, with t[0] = 0
            
            Inputs - 
                unknowns      - data dictionary of segment free unknowns
                conditions    - data dictionary of segment conditions
                differentials - data dictionary of non-dimensional differential operators
                
            Outputs - 
                differentials - udpated data dictionary with dimensional differentials 
            
            Assumptions - 
                outputed operators are in dimensional time for the current solver iteration
                Base_Segment() by default will rescale operators based on final time,
                    either found in 
                    segment.unknowns.finals.time (will update segment.conditions.inertial.time)
                        otherwise,
                    segment.conditions.inertial.time[-1] - segment.conditions.inertial.time[0]
                
        """
        
        # unpack
        t = differentials.t
        D = differentials.D
        I = differentials.I
        
        # rescale time
        if unknowns.finals.has_key('time'):
            # variable final time
            dt = unknowns.finals.time
            t = t * dt
            # update inertial time, keep start time
            conditions.inertial.time = t + conditions.inertial.time[0]
        else:
            # stationary time control points
            dt = conditions.inertial.time[-1] - segment.conditions.inertial.time[0]
            t = t * dt
        
        # rescale operators
        D = D / dt
        I = I * dt
        
        # pack
        differentials.t = t
        differentials.D = D
        differentials.I = I
        
        return differentials


# ----------------------------------------------------------------------
#  Handle Linking
# ----------------------------------------------------------------------

class Container(ContainerBase):
    pass

Base_Segment.Container = Container
