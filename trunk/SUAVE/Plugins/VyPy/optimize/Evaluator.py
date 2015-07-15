
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from VyPy.data import Object
        
# ----------------------------------------------------------------------
#   Evaluator
# ----------------------------------------------------------------------
class Evaluator(object):
    
    def __init__(self,function=None,gradient=None,hessian=None):
        self.function = function or self.function
        self.gradient = gradient or self.gradient
        self.hessian  = hessian  or self.hessian
        
        # check if methods were implemented
        for meth in ['function','gradient','hessian']:
            this_method = getattr(self,meth)
            base_method = getattr(Evaluator,meth)
            
            # method was not overriden, set to None
            try:
                if this_method.__func__ is base_method.__func__:
                    setattr(self,meth,None)
            except AttributeError:
                pass
            
    
    def function(self,variables):
        """ outputs = Evaluator.function(variables)
        
            Example Snippet:
            
            def function(self,variables):
                try:
                    # code
                    #
                    #
                    outputs = {}
                except Exception:
                    outputs = {}
                return outputs 
        """
        raise NotImplementedError
    
    def gradient(self,variables):
        """ outputs = Evaluator.gradient(variables)
        
            Example Snippet:
            
            def gradient(self,variables):
                try:
                    # code
                    #
                    #
                    outputs = {}
                except Exception:
                    outputs = {}
                return outputs 
        """
        raise NotImplementedError
    
    def hessian(self,variables):
        """ outputs = Evaluator.hessian(variables)
        
            Example Snippet:
            
            def hessian(self,variables):
                try:
                    # code
                    #
                    #
                    outputs = {}
                except Exception:
                    outputs = {}
                return outputs 
        """
        raise NotImplementedError
