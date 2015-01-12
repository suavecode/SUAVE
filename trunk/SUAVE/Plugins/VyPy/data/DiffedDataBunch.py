
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from DataBunch import DataBunch

from copy import deepcopy

# ----------------------------------------------------------------------
#  DiffedDataBunch
# ----------------------------------------------------------------------

class DiffedDataBunch(DataBunch):
    """ VyPy.data.DiffedDataBunch()
    """
    
    def __defaults__(self):
        self.tag    = 'config'
        self._base  = DataBunch()
        self._diff  = DataBunch()
        
    def __init__(self,base=None):
        if base is None: base = DataBunch()
        self._base = base
        this = deepcopy(base)
        DataBunch.__init__(self,this)
        
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
            return DataBunch.__str__(self,indent)

            
# ------------------------------------------------------------
#  Diffing Function
# ------------------------------------------------------------

def diff(A,B):
    
    keys = set([])
    keys.update( A.keys() )
    keys.update( B.keys() )
    
    if isinstance(A,DiffedDataBunch):
        keys.remove('_base')
        keys.remove('_diff')
    
    result = type(A)()
    result.clear()
    
    for key in keys:
        va = A.get(key,None)
        vb = B.get(key,None)
        if isinstance(va,DataBunch) and isinstance(vb,DataBunch):
            sub_diff = diff(va,vb)
            if sub_diff:
                result[key] = sub_diff
        elif (isinstance(va,DataBunch) or isinstance(vb,DataBunch)) or not va==vb:
            result[key] = va
        
    return result    