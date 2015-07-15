
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from Objective import Objective
from Evaluator import Evaluator

import VyPy
from VyPy.data import Object, IndexableDict, Descriptor
from VyPy.data.input_output import flatten_list
from VyPy.tools import atleast_2d_col, atleast_2d_row

# ----------------------------------------------------------------------
#   Constraint Function
# ----------------------------------------------------------------------

class Constraint(Objective):
    
    Container = None
    
    def __init__( self, evaluator=None, 
                  tag='c', sense='=', edge=0.0, 
                  scale=1.0,
                  variables=None):
        
        Objective.__init__(self)
        
        self.evaluator = evaluator
        self.tag       = tag
        self.sense     = sense
        self.edge      = edge
        self.scale     = scale
        self.variables = variables
        

    def __check__(self):

        Objective.__check__(self)
            
        # arrays
        self.edge = atleast_2d_col(self.edge)
        if not isinstance(self.scale,VyPy.data.scaling.ScalingFunction):
            self.scale = atleast_2d_col(self.scale)
            
    def __repr__(self):
        return "<Constraint '%s'>" % self.tag        
    
    
# ----------------------------------------------------------------------
#   Constraint Container
# ----------------------------------------------------------------------

from Equality   import Equality, Equalities
from Inequality import Inequality, Inequalities

class_map = {
    '=' : Equality ,
    '>' : Inequality,
    '<' : Inequality,
}

class Constraints(Object):
    
    def __init__(self,variables):
        self.variables    = variables
        self.equalities   = Equalities(self.variables)
        self.inequalities = Inequalities(self.variables)
        
        self._container_map = {
            '=' : self.equalities ,
            '>' : self.inequalities,
            '<' : self.inequalities,
        }               
    
    def __set__(self,problem,arg_list):            
        self.clear()
        self.extend(arg_list)
        
    def clear(self):
        self.equalities.clear()
        self.inequalities.clear()
    
    def append(self, evaluator, 
               tag=None, sense='=', edge=0.0, 
               scale=1.0 ):
        
        if type(evaluator) is Constraint:
            constraint = evaluator
            evaluator = constraint.evaluator
            tag   = constraint.tag
            sense = constraint.sense
            edge  = constraint.edge
            scale = constraint.scale
            
        if tag is None and isinstance(evaluator,Constraint):
            constraint = evaluator
            constraint.variables = self.variables
        else:
            constraint = class_map[sense](evaluator,tag,sense,edge,scale,self.variables)
        
        constraint.__check__()
        
        sense = constraint.sense
        if sense not in class_map.keys():
            raise KeyError , 'invalid constraint sense "%s"' % sense        
        
        tag = constraint.tag
        self._container_map[sense][tag] = constraint
        
    def extend(self,arg_list):
        for args in arg_list:
            args = flatten_list(args)
            self.append(*args)
            
    def tags(self):
        return self.equalities.tags() + self.inequalities.tags()
    def senses(self):
        return self.equalities.senses() + self.inequalities.senses()
    def edges(self):
        return self.equalities.edges() + self.inequalities.edges()
    def scales(self):
        return self.equalities.scales() + self.inequalities.scales()
    def evaluators(self):
        return self.equalities.evaluators() + self.inequalities.evaluators()
    
    # no set(), modify constraints by type
    
    def __len__(self):
        return len(self.equalities) + len(self.inequalities)
    def items(self):
        return self.equalities.items() + self.inequalities.items()
    def __repr__(self):
        return repr(self.equalities) + '\n' + repr(self.inequalities)
    def __str__(self):
            return str(self.equalities) + '\n' + str(self.inequalities)

# ----------------------------------------------------------------------
#   Constraint Function
# ----------------------------------------------------------------------
Constraint.Container = Constraints