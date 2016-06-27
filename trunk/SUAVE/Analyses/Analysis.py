# Analysis.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
from SUAVE.Core import Container as ContainerBase
from SUAVE.Core import Results


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
    
    def compile(self,*args,**kwarg):
        """ compile the data, settings, etc.
            avoid analysis specific algorithms
        """
        return
        
    def initialize(self,*args,**kwarg):
        """ analysis specific initialization algorithms
        """
        return
    
    def evaluate(self,*args,**kwarg):
        """ analysis specific evaluation algorithms
        """        
        raise NotImplementedError
        return Results()
    
    def finalize(self,*args,**kwarg):
        """ analysis specific finalization algorithms
        """        
        return 
    
    def __call__(self,*args,**kwarg):
        return self.evaluate(*args,**kwarg)
    

# ----------------------------------------------------------------------
#  Config Container
# ----------------------------------------------------------------------

class Container(ContainerBase):
    """ SUAVE.Analyses.Analysis.Container()
    """
    
    def compile(self,*args,**kwarg):
        for tag,analysis in self.items():
            if hasattr(analysis,'compile'):
                analysis.compile(*args,**kwarg)
        
    def initialize(self,*args,**kwarg):
        for tag,analysis in self.items:
            if hasattr(analysis,'initialize'):
                analysis.initialize(*args,**kwarg)
    
    def evaluate(self,*args,**kwarg):
        results = Results()
        for tag,analysis in self.items(): 
            if hasattr(analysis,'evaluate'):
                result = analysis.evaluate(*args,**kwarg)
            else:
                result = analysis(*args,**kwarg)
            results[tag] = result
        return results
    
    def finalize(self,*args,**kwarg):
        for tag,analysis in self.items():
            if hasattr(analysis,'finalize'):
                analysis.finalize(*args,**kwarg)
    
    def __call__(self,*args,**kwarg):
        return self.evaluate(*args,**kwarg)


# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------

Analysis.Container = Container