
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------        

from VyPy.data import ibunch, obunch

# ----------------------------------------------------------------------
#   Driver
# ----------------------------------------------------------------------        

class Driver(object):
    
    def __init__(self):
        
        self.verbose = True
        self.other_options = obunch()
    
    def run(self,problem):
        raise NotImplementedError
    
    def pack_outputs(self,vars_min):
        
        # unpack
        objectives = self.problem.objectives
        equalities = self.problem.constraints.equalities
        inequalities = self.problem.constraints.inequalities
        
        # start the data structure
        outputs = ibunch()
        outputs.variables    = None
        outputs.objectives   = ibunch()
        outputs.equalities   = ibunch()
        outputs.inequalities = ibunch()
        outputs.success      = False
        outputs.messages     = ibunch()
        
                
        # varaiables
        outputs.variables = vars_min
        
        # objectives
        for tag in objectives.tags():
            outputs.objectives[tag] = objectives[tag].evaluator.function(vars_min)[tag]
        
        # equalities
        for tag in equalities.tags():
            outputs.equalities[tag] = equalities[tag].evaluator.function(vars_min)[tag]
            
        # inequalities
        for tag in inequalities.tags():
            outputs.inequalities[tag] = inequalities[tag].evaluator.function(vars_min)[tag]
            
        return outputs