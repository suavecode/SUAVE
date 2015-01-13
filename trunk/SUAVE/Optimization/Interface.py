

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE

from SUAVE.Core import Data, Data_Exception, Data_Warning
from SUAVE.Core import Container as ContainerBase


# ----------------------------------------------------------------------
#  Config
# ----------------------------------------------------------------------

class Interface(Data):
    """ SUAVE.Optimization.Interface()
    """
    def __defaults__(self):
        self.tag    = 'interface'
        
        self.inputs   = Data()
        self.configs  = SUAVE.Components.Configs.Config.Container()
        self.analyses = SUAVE.Analyses.Analysis.Container()
        self.process  = SUAVE.Analyses.Process.Container()
        self.results  = SUAVE.Analyses.Results()
        
        
    def evaluate(self,inputs):
        self.inputs = inputs
        interface = self
        
        for key,step in self.process.items():
            if hasattr(step,'evaluate'):
                result = step.evaluate(interface)
            else:
                result = step(interface)
            self.results[key] = result
            
            
        return self.results
        
        
        
## ----------------------------------------------------------------------
##   Strategy
## ----------------------------------------------------------------------
        
#import types        
        
#class Strategy(Data):
    
    #def __setattr__(self,tag,value):
        #self.append_step(value,tag)
    
    #def append_step(self,step,tag=None):
        
        #if tag is None:
            #tag = step.tag
        
        #step = types.MethodType(step,self)
        #Data.__setattr__(self,tag,step)

# ----------------------------------------------------------------------
#  Config Container
# ----------------------------------------------------------------------

class Container(ContainerBase):
    """ SUAVE.Optimization.Interface.Container()
    """
    pass


# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------

Interface.Container = Container