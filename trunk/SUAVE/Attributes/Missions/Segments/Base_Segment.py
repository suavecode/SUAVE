
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


# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

class Base_Segment(Data):
    
    # ------------------------------------------------------------------
    #   Methods For Overriding
    # ------------------------------------------------------------------    

    def __defaults__(self):
        self.tag = 'Base_Segment'
        
        # --- Conditions and Unknowns
        
        # user shouldn't change these in an input script
        # only used for processing / post processing
        # they will be shared with analysis modules and meaningful naming is important
        
        # base array column lengths
        # use a trivial operation to copy the array
        ones_1col = np.zeros([1,1])
        ones_2col = np.zeros([1,2])
        ones_3col = np.zeros([1,3])         
        
        
        # --- Conditions 
        
        # setup conditions
        conditions = Data()
        conditions.inertial = Data()
        conditions.body     = Data()
        conditions.weights  = Data()
        conditions.engergy  = Data()   
        
        # inertial conditions
        conditions.inertial.position_vector      = ones_3col + 0
        conditions.inertial.velocity_vector      = ones_3col + 0
        conditions.inertial.acceleration_vector  = ones_3col + 0
        conditions.inertial.total_force_vector   = ones_3col + 0      
        conditions.inertial.time                 = ones_1col + 0
        
        # body conditions
        conditions.body.velocity_vector          = ones_3col + 0
        conditions.body.acceleration_vector      = ones_3col + 0
        conditions.body.total_force_vector       = ones_3col + 0
        
        # weights conditions
        conditions.weights.total_mass            = ones_1col + 0
        
        # energy conditions
        conditions.energies.total_energy         = ones_1col + 0
        conditions.energies.total_efficiency     = ones_1col + 0
        
        # --- Unknowns
        
        # setup unknowns
        unknowns = Data()
        unknowns.states   = Data()
        unknowns.controls = Data()
        unknowns.finals   = Data()
        
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
        self.differentials.t = t
        self.differentials.D = D
        self.differentials.I = I
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
                conditions - data dictionary of update conditions
                
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
                conditions - data dictionary with additional post-processed data
            
            Usage Notes - 
                use this to store the unknowns and any other interesting in conditions
                    for later plotting
            
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
    #   Probably Internal Methods
    # ------------------------------------------------------------------

    def initialize_arrays(self,unknowns, conditions):
        """ Segment.initialize_arrays(unknowns, conditions)
            expands the number of rows of all arrays in segment.conditions
            and segment.unknowns to segment.options.n_control_points
            
            Inputs - 
                unknowns   - data dicitonary of segment unknowns
                conditions - data dictionary of segment conditions
                
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
         
        return unknowns, conditions
    
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
    



def segment_residuals(x,segment):
    """ segment_residuals(x)
        the objective function for the segment solver
        
        Inputs - 
            x - 1D vector of the solver's guess for segment free unknowns
        
        Outputs - 
            R - 1D vector of the segment's residuals
            
        Assumptions -
            solver tries to converge R = [0]
            
    """
    
    # unpack segment
    unknowns      = segment.unknowns
    conditions    = segment.conditions
    differentials = deepcopy( segment.differentials )
    
    # unpack vector into unknowns
    unknowns.unpack_array(x)
    
    # update differentials
    differentials = segment.update_differentials(unknowns,conditions,differentials)
    t = differentials.t
    D = differentials.D
    I = differentials.I
    
    # update conditions
    conditions = segment.update_conditions(unknowns,conditions,differentials)
    
    # solve residuals
    residuals = segment.solve_residuals(unknowns,conditions,differentials)
    
    # pack column matrices
    S  = unknowns .states  .pack_array()
    FS = residuals.states  .pack_array()
    FC = residuals.controls.pack_array()
    FF = residuals.finals  .pack_array()
    
    # solve final residuals
    R = [ ( np.dot(D,S) - FS ) ,
          (               FC ) , 
          (               FF )  ]
    
    # pack in to final residual vector
    R = np.hstack( [ r.ravel(order='F') for r in R ] )
    
    return R


# ----------------------------------------------------------------------
#  Handle Linking
# ----------------------------------------------------------------------

class Container(ContainerBase):
    pass

Base_Segment.Container = Container
