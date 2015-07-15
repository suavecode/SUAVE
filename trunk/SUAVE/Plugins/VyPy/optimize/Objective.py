
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from Evaluator import Evaluator

from VyPy.data import IndexableDict
from VyPy.tools.arrays import atleast_2d_col, atleast_2d_row

import numpy as np
    
# ----------------------------------------------------------------------
#   Objective Function
# ----------------------------------------------------------------------

class Objective(Evaluator):
    
    Container = None
    
    # TODO: tag=None
    def __init__(self,evaluator=None,tag='f',scale=1.0,variables=None,):
        
        Evaluator.__init__(self)
        
        self.evaluator = evaluator
        self.tag       = tag
        self.scale     = scale
        self.variables = variables
        
    def __check__(self):
        
        if not isinstance(self.evaluator, Evaluator):
            self.evaluator = Evaluator(function=self.evaluator)        
        
        if self.evaluator.gradient is None:
            self.gradient = None
        if self.evaluator.hessian is None:
            self.hessian = None
        
    def function(self,x):
        
        x = self.variables.scaled.unpack_array(x)
        
        func = self.evaluator.function
        tag  = self.tag
        scl  = self.scale
        
        result = func(x)[tag]
        
        result = atleast_2d_col(result)
        
        result = result / scl
        
        return result
    
    def gradient(self,x):
        
        x = self.variables.scaled.unpack_array(x)
        
        func = self.evaluator.gradient
        tag  = self.tag
        fscl = self.scale
        
        res = func(x)[tag]
        
        result = [ atleast_2d_row(res[k]) * self.variables[k].scale 
                   for k in self.variables.keys() ]
        result = np.hstack(result)
        
        result = result / fscl ## !!! PROBLEM WHEN SCL is NOT CENTERED
        
        return result
    
    def hessian(self,x):
        raise NotImplementedError
    
    def __repr__(self):
        return "<Objective '%s'>" % self.tag
    
    
# ----------------------------------------------------------------------
#   Objectives Container
# ----------------------------------------------------------------------

class Objectives(IndexableDict):
    
    def __init__(self,variables):
        self.variables = variables
    
    def __set__(self,problem,arg_list):            
        self.clear()
        self.extend(arg_list)
    
    def append(self,evaluator,tag=None,scale=1.0):
        if tag is None and isinstance(evaluator,Objective):
            objective = evaluator
            objective.variables = self.variables
        else:
            objective = Objective(evaluator,tag,scale,self.variables)
        
        objective.__check__()
        tag = objective.tag
        self[tag] = objective
        
    def extend(self,arg_list):
        for args in arg_list:
            self.append(*args)
        
    def tags(self):
        return self.keys()
    def scales(self):
        return [ obj.scale for obj in self.values() ]
    def evaluators(self):
        return [ obj.evaluator for obj in self.values() ]
    
    def set(self,scales=None):
        if scales:
            for i,s in enumerate(scales):
                self[i].scale = s
                
# ----------------------------------------------------------------------
#   Handle Linking
# ----------------------------------------------------------------------
Objective.Container = Objectives

