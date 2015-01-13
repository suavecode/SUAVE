

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data, Data_Exception, Data_Warning
from SUAVE.Core import Container as ContainerBase
from Results import Results


# ----------------------------------------------------------------------
#  Process
# ----------------------------------------------------------------------

class Process(Data):
    
    def __defaults__(self):
        self.tag = 'process'
        
    def evaluate(self,interface):
        return Results()
    
    #def finalize(self):
        #return
    
    
# ----------------------------------------------------------------------
#  Config Container
# ----------------------------------------------------------------------

class Container(ContainerBase):
    """ SUAVE.Analyses.Analysis.Container()
    """
    
    def evaluate(self,interface):
        results = Results()
        for tag,process in self.items(): 
            #if not callable(process): continue
            result = process(interface)
            results[tag] = result
        return results
    
    #def finalize(self):
        #for analysis in self:
            #if hasattr(analysis,'finalize'):
                #analysis.finalize()  
    
    __call__ = evaluate


#import inspect
#def get_args(obj):
    #if hasattr(obj,'__call__'):
        #return inspect.getargspec(obj.__call__).args
    #else:
        #return inspect.getargspec(obj).args
    

# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------

Process.Container = Container