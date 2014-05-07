
# helper class for implemeting syntax sugar
class Descriptor(object):
    def __init__(self,key):
        self._key = key
    def __get__(self,obj,kls=None):
        return getattr(obj,self._key)
    def __set__(self,obj,val):
        getattr(obj,self._key).__set__(obj,val)