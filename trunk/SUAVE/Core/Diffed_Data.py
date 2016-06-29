# DiffedData.py
#
# Created:  Feb 2015, T. Lukacyzk
# Modified: Feb 2016, T. MacDonald
#           Jun 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from copy import deepcopy
from Container import Container as ContainerBase
from Data import Data
import numpy as np

# ----------------------------------------------------------------------
#  Config
# ----------------------------------------------------------------------

class Diffed_Data(Data):
    """ SUAVE.Core.DiffedData()
    """
    
    def __defaults__(self):
        self.tag    = 'config'
        self._base  = Data()
        self._diff  = Data()
        
    def __init__(self,base=None):
        if base is None: base = Data()
        self._base = base
        this = deepcopy(base) # deepcopy is needed here to build configs - Feb 2016, T. MacDonald
        Data.__init__(self,this)
        
    def store_diff(self):
        delta = diff(self,self._base)
        self._diff = delta
        
    def pull_base(self):
        try: self._base.pull_base()
        except AttributeError: pass
        self.update(self._base)
        self.update(self._diff)
    
    def __str__(self,indent=''):
        try: 
            args = self._diff.__str__(indent)
            args += indent + '_base : ' + self._base.__repr__() + '\n'
            args += indent + '  tag : ' + self._base.tag + '\n'
            return args
        except AttributeError:     
            return Data.__str__(self,indent)
    
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

# ------------------------------------------------------------
#  Diffing Function
# ------------------------------------------------------------

def diff(A,B):

    keys = set([])
    keys.update( A.keys() )
    keys.update( B.keys() )

    if isinstance(A,Diffed_Data):
        keys.remove('_base')
        keys.remove('_diff')

    result = type(A)()
    result.clear()

    for key in keys:
        va = A.get(key,None)
        vb = B.get(key,None)
        if isinstance(va,Data) and isinstance(vb,Data):
            sub_diff = diff(va,vb)
            if sub_diff:
                result[key] = sub_diff

        elif isinstance(va,Data) or isinstance(vb,Data):
            result[key] = va

        elif not np.all(va == vb):
            result[key] = va

    return result    