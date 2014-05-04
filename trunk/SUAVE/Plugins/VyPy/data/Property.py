
# objects hidden in the dictionary
class Property(object):
    def __init__(self,key=None):
        self._key = key
    def __get__(self,obj,kls=None):
        if obj is None: return self
        else          : return dict.__getitem__(obj,self._key)
    def __set__(self,obj,val):
        dict.__setitem__(obj,self._key,val)
    def __delete__(self,obj):
        dict.__delitem__(obj,self._key)

