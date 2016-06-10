# DiffedData.py
#
# Created:  Feb 2015, T. Lukacyzk
# Modified: Feb 2016, T. MacDonald
#           Jun 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from DiffedDataBunch import DiffedDataBunch
from Container import Container as ContainerBase
from Data import Data

# ----------------------------------------------------------------------
#  Config
# ----------------------------------------------------------------------

class Diffed_Data(DiffedDataBunch,Data):
    """ SUAVE.Core.DiffedData()
    """
    
    def finalize(self):
        ## dont do this here, breaks down stream dependencies
        # self.store_diff 
        
        self.pull_base()

# ----------------------------------------------------------------------
#  Config Container
# ----------------------------------------------------------------------

class Container(ContainerBase):
    """ SUAVE.Core.Diffed_Data.Container()
    """
    def append(self,value):
        try: value.store_diff()
        except AttributeError: pass
        ContainerBase.append(self,value)
        
    def pull_base(self):
        for config in self:
            try: config.pull_base()
            except AttributeError: pass

    def store_diff(self):
        for config in self:
            try: config.store_diff()
            except AttributeError: pass
    
    def finalize(self):
        for config in self:
            try: config.finalize()
            except AttributeError: pass


# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------

Diffed_Data.Container = Container
