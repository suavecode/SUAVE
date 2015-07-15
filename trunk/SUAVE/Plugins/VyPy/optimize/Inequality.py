
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from Evaluator  import Evaluator
from Constraint import Constraint

from VyPy.data import IndexableDict
from VyPy.data.input_output import flatten_list
from VyPy.tools.arrays import atleast_2d, atleast_2d_row

    
# ----------------------------------------------------------------------
#   Inequality Function
# ----------------------------------------------------------------------

class Inequality(Constraint):
    
    Container = None
    
    def __init__( self, evaluator=None, 
                  tag='cieq', sense='<', edge=0.0, 
                  scale=1.0,
                  variables=None):
        Constraint.__init__(self,evaluator,
                            tag,sense,edge,
                            scale,variables)
        
    def function(self,x):
        
        snz  = self.sense
        edg  = self.edge
        scl  = self.scale
        
        result = Constraint.function(self,x)
        
        if snz == '>':
            result = edg/scl - result
        elif snz == '<':
            result = result - edg/scl
        else:
            raise Exception, 'unrecognized sense %s' % snz        
        
        return result
    
    def gradient(self,x):
        
        snz  = self.sense
        
        result = Constraint.gradient(self,x)
        
        if snz == '>':
            result = -1 * result
        elif snz == '<':
            result = +1 * result
        else:
            raise Exception, 'unrecognized sense %s' % snz        
        
        return result 
    
    def hessian(self,x):
        raise NotImplementedError

    def __repr__(self):
        return "<Inequality '%s'>" % self.tag
    

# ----------------------------------------------------------------------
#   Inequality Container
# ----------------------------------------------------------------------

class Inequalities(IndexableDict):
    
    def __init__(self,variables):
        self.variables = variables
    
    def __set__(self,problem,arg_list):            
        self.clear()
        self.extend(arg_list)
        
    def append(self, evaluator, 
               tag=None, sense='=', edge=0.0, 
               scale=1.0 ):
        
        if tag is None and isinstance(evaluator,Inequality):
            inequality = evaluator
            inequality.variables = self.variables
        else:
            args = flatten_list(args) + [self.variables]
            inequality = Inequality(evaluator,tag,sense,edge,scale,self.variables)
        
        inequality.__check__()
        tag = inequality.tag
        self[tag] = inequality
        
    def extend(self,arg_list):
        for args in arg_list:
            self.append(*args)
                        
    def tags(self):
        return self.keys()
    def senses(self):
        return [ con.sense for con in self.values() ]
    def edges(self):
        return [ con.edge for con in self.values() ]
    def scales(self):
        return [ con.scale for con in self.values() ]
    def evaluators(self):
        return [ con.evaluator for con in self.values() ]
    
    def edges_array(self):
        return np.vstack([ atleast_2d(x,'col') for x in self.edges() ])       
    
    def set(senses=None,edges=None,scales=None):
        if senses:
            for i,s in enumerate(senses):
                self[i].sense = s            
        if edges:
            for i,e in enumerate(edges):
                self[i].edge = e
        if scales:
            for i,s in enumerate(scales):
                self[i].scale = s       
                
# ----------------------------------------------------------------------
#   Inequality Container
# ----------------------------------------------------------------------
Inequality.Container = Inequalities
