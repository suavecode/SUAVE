

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Container
from Results import Results


# ----------------------------------------------------------------------
#  Process
# ----------------------------------------------------------------------

class Process(Container):
    
    def evaluate(self,*args,**kwarg):
        
        results = Results()
        
        for tag,step in self.items(): 
            
            #if not callable(step): continue
            
            if hasattr(step,'evaluate'): 
                result = step.evaluate(*args,**kwarg)
            else:
                result = step(*args,**kwarg)
                
            results[tag] = result
        
        #: for each step
        
        return results
        
    def __call__(self,*args,**kwarg):
        return self.evaluate(*args,**kwarg) 
    
    
#import inspect
#def get_args(obj):
    #if hasattr(obj,'__call__'):
        #return inspect.getargspec(obj.__call__).args
    #else:
        #return inspect.getargspec(obj).args
    
