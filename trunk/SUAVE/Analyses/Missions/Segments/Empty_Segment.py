
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# python imports
import numpy as np
from copy import deepcopy

# SUAVE imports
from Base_Segment                       import Base_Segment
from SUAVE.Structure                    import Data, Data_Exception
from SUAVE.Structure                    import Container as ContainerBase
from SUAVE.Methods.Utilities.Chebyshev  import chebyshev_data
from SUAVE.Methods.Utilities            import atleast_2d_col

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

class Empty_Segment(Base_Segment):
    
    # ------------------------------------------------------------------
    #   Data Defaults
    # ------------------------------------------------------------------    

    def __defaults__(self):
        self.tag = 'Empty_Segment'
        
        # --- Segment Inputs
        
        # these are the inputs the user will define in the input script
        
        ## EXAMPLE
        #self.mach_number = 0.7
                
        
        # --- Conditions and Unknowns
        
        # used for processing / post processing
        # they will be shared with analysis modules and meaningful naming is important
        # a user shouldn't change these in an input script
        
        # base array column lengths
        # use a trivial operation to copy the array
        ones_1col = np.ones([1,1])
        ones_2col = np.ones([1,2])
        ones_3col = np.ones([1,3])         
        
        
        # --- Conditions 
        
        # setup conditions
        conditions = Data()
        self.conditions = conditions
        
        ## EXAMPLE
        #conditions.frames   = Data()
        #conditions.frames.inertial = Data()        
        #conditions.frames.inertial.position_vector = ones_3col * 0
        
        
        # --- Unknowns
        
        # setup unknowns
        unknowns = Data()
        unknowns.states   = Data()
        unknowns.controls = Data()
        unknowns.finals   = Data()
        self.unknowns = unknowns
        
        ## EXAMPLE
        # unknowns.states.gamma = ones_1col + 0

        
        # --- Residuals
         
        # setup unknowns
        # must implement the same fields as unknowns.(...)
        residuals = Data()
        residuals.states   = Data()
        residuals.controls = Data()
        residuals.finals   = Data()
        self.residuals = residuals
        
        ## EXAMPLE
        # unknowns.states.gamma = ones_1col + 0
        
        
        return

    # ------------------------------------------------------------------
    #   Methods For Initialization
    # ------------------------------------------------------------------  
    
    def check_inputs(self):
        """ Segment.check():
            error checking of segment inputs
            assume no segment work has been done yet
        """
        
        # unpack inputs
        ## CODE
        
        # check inputs
        ## if problem: raise Exception
        
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
                numerics - data dictionary of dimensional differential operators
                
            Outputs:
                conditions - the conditions data dictionary, updated with the 
                             values that can be precalculated
            
            Assumptions:
                preserves the shapes of arrays in conditions
                
        """
        
        # Usage Notes:
        #     this function will be called before solving the segment's free unknowns
        #     may need to inspect segment (self) for user inputs
        #     arrays will be expanded to number of control points in numerics.n_control_points
        
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
    
    def update_conditions(self,conditions,numerics,unknowns):
        """ Segment.update_conditions(conditions,numerics,unknowns)
            if needed, updates the conditions given the current free unknowns and numerics
            called once per segment solver iteration
            
            Inputs - 
                unknowns      - data dictionary of segment free unknowns with fields:
                    states, controls, finals
                    these are defined in segment.__defaults__
                numerics - data dictionary of differential operators for this iteration
                conditions    - data dictionary of segment conditions
                    these are defined in segment.__defaults__
                
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
            solves the residuals for the free unknowns, called once per segment solver iteration
            
            Inputs - 
                unknowns   - data dictionary of segment free unknowns with fields:
                    states, controls, finals
                    these are defined in segment.__defaults__
                conditions - data dictionary of segment conditions
                    these are defined in segment.__defaults__
                numerics   - data dictionary of differential operators for this iteration
                
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
                numerics - data dictionary of the converged differential operators
                conditions - data dictionary of segment conditions
                    these are defined in segment.__defaults__
                
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
    
    #def initialize_differentials(self,numerics):
    #   see Base_Segment()
    
    #def update_differentials(self,conditions,numerics,unknowns=None):
    #   see Base Segment()



# ----------------------------------------------------------------------
#  Handle Linking
# ----------------------------------------------------------------------

class Container(ContainerBase):
    pass

Base_Segment.Container = Container

# ----------------------------------------------------------------------
#  Module Tests
# ----------------------------------------------------------------------

if __name__ == '__main__':
    raise NotImplementedError
    
