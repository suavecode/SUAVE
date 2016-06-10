# Data.py
#
# Created:  Jan 2015, T. Lukacyzk
# Modified: Feb 2016, T. MacDonald

""" SUAVE Data Base Classes
"""

from collections import OrderedDict

# for enforcing attribute style access names
import string
chars = string.punctuation + string.whitespace
t_table = string.maketrans( chars          + string.uppercase , 
                            '_'*len(chars) + string.lowercase )

#from warnings import warn

# ----------------------------------------------------------------------
#   Data
# ----------------------------------------------------------------------        

class Data_Base(OrderedDict):
    
    def __defaults__(self):
        return 
    
    #def __repr__(self, _repr_running={}):
        #"""od.__repr__() <==> repr(od)"""
        #call_key = id(self), _get_ident()
        #if call_key in _repr_running:
            #return '...'
        #_repr_running[call_key] = 1
        #try:
            #if not self:
                #return '%s()' % (self.__class__.__name__,)
            #return '%s(%r)' % (self.__class__.__name__, self.items())
        #finally:
            #del _repr_running[call_key]
            
    def __repr__(self):
        """ Invertible* string-form of a Dict.
        """
        keys = self.keys()
        args = ', '.join(['%s=%r' % (key, self[key]) for key in keys if not key.startswith('_')])
        return '%s(%s)' % (self.__class__.__name__, args)    

