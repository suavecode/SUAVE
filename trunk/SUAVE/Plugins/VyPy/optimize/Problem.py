
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from Variable   import Variable, Variables
from Objective  import Objective, Objectives
from Constraint import Constraint, Constraints
from Equality   import Equality
from Inequality import Inequality

from VyPy.data import IndexableDict, Object, Descriptor


# ----------------------------------------------------------------------
#   Problem
# ----------------------------------------------------------------------

class Problem(Object):
    
    def __init__(self):
        
        self.variables    = Variables()
        self.objectives   = Objectives(self.variables)
        self.constraints  = Constraints(self.variables)
        self.equalities   = self.constraints.equalities
        self.inequalities = self.constraints.inequalities
      
    def has_gradients(self):
        
        # objectives
        grads = [ not evalr.gradient is None for evalr in self.objectives ]
        obj_grads = any(grads) and all(grads)
        
        # inequalities
        grads = [ not evalr.gradient is None for evalr in self.inequalities ]
        ineq_grads = any(grads) and all(grads)
            
        # equalities
        grads = [ not evalr.gradient is None for evalr in self.equalities ]
        eq_grads = any(grads) and all(grads)            
           
        return obj_grads, ineq_grads, eq_grads

  
