
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# numpy imports
import numpy as np
from scipy.optimize import root
from copy import deepcopy

# SUAVE imports
from SUAVE.Structure           import Data
from SUAVE.Methods.Solvers     import jacobian_complex


# ----------------------------------------------------------------------
#  Evaluate a Segment
# ----------------------------------------------------------------------
        
def evaluate_segment(segment):
    """  solution = evaluate_segment(segment)
         integrate a Segment 
         
         Inputs:
         
         Outputs:
         
         Assumptions:
         
    """
    
    # check inputs
    segment.check_inputs()
    
    # unpack segment
    options       = segment.options
    unknowns      = segment.unknowns    
    conditions    = segment.conditions
    residuals     = segment.residuals
    differentials = segment.differentials
    initials      = segment.initials
    
    # initialize arrays
    unknowns,conditions,residuals = segment.initialize_arrays(unknowns,conditions,residuals,options)
    
    # initialize differential operators
    differentials = segment.initialize_differentials(differentials,options)

    # preprocess segment conditions
    conditions = segment.initialize_conditions(conditions,differentials,initials)
    
    # pack the guess
    guess = unknowns.pack_array('vector')

    # solve system
    x_sol = root( fun    = segment_residuals          ,
                  x0     = guess                      ,
                  args   = segment                    ,
                  method = "hybr"                     ,
                  #jac    = jacobian_complex           ,
                  tol    = options.tolerance_solution  )
    
    # confirm final solution
    segment_residuals(x_sol.x,segment)
    unknowns      = segment.unknowns    
    conditions    = segment.conditions
    differentials = segment.differentials
    
    # update differential operators
    differentials = segment.update_differentials(unknowns,conditions,differentials)    

    # post processing
    segment.post_process(unknowns,conditions,differentials)
    
    # done!
    return segment

   
# ----------------------------------------------------------------------
#  Main Segment Objective Function
# ----------------------------------------------------------------------

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
    residuals     = segment.residuals
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
    residuals = segment.solve_residuals(unknowns,residuals,conditions,differentials)
    
    # pack column matrices
    S  = unknowns .states  .pack_array('array')
    FS = residuals.states  .pack_array('array')
    FC = residuals.controls.pack_array('array')
    FF = residuals.finals  .pack_array('array')
    
    if S:
        DFS = np.dot(D,S)
    else:
        DFS = np.array([[]])
    
    # solve final residuals
    R = [ ( DFS - FS ) ,
          (       FC ) , 
          (       FF )  ]
    
    # pack in to final residual vector
    R = np.hstack( [ r.ravel(order='F') for r in R ] )
    
    return R
