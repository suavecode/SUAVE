# Process.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import ContainerOrdered
from SUAVE.Analyses.Results import Results

# ----------------------------------------------------------------------
#  Process
# ----------------------------------------------------------------------

class Process(ContainerOrdered):
    
    verbose = False
    
    def evaluate(self,*args,**kwarg):
        
        results = Results()
        
        if self.verbose:
            print 'process start'
        
        for tag,step in self.items(): 
            
            if self.verbose:
                print 'step :' , tag
            
            #if not callable(step): continue
            
            if hasattr(step,'evaluate'): 
                result = step.evaluate(*args,**kwarg)
            else:
                result = step(*args,**kwarg)
                
            results[tag] = result
        
        #: for each step
        
        if self.verbose:
            print 'process end'        
        
        return results
        
    def __call__(self,*args,**kwarg):
        return self.evaluate(*args,**kwarg) 
    
