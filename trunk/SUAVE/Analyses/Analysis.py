

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data, Data_Exception, Data_Warning
from SUAVE.Structure import Container as ContainerBase
from Results import Results


# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

class Analysis(Data):
    """ SUAVE.Analyses.Analysis()
    """
    def __defaults__(self):
        self.tag    = 'analysis'
        self.features = Data()
        self.settings = Data()
        
        
    def evaluate(self,condtitions):
        return Results()
    
    def finalize(self):
        return 
    
    __call__ = evaluate
        

# ----------------------------------------------------------------------
#  Config Container
# ----------------------------------------------------------------------

class Container(ContainerBase):
    """ SUAVE.Analyses.Analysis.Container()
    """
    
    def evaluate(self,condtitions):
        results = Results()
        
        for tag,analysis in self.items(): 
            #if not callable(analysis): continue
            
            result = analysis(conditions)
            results[tag] = result
                
        return results
    
    def finalize(self):
        for analysis in self:
            try: analysis.finalize()
            except AttributeError: pass    
    
    __call__ = evaluate


# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------

Analysis.Container = Container