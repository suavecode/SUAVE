# IndexableDict.py
#
# Created:  Aug 2015, T. Lukacyzk
# Modified: Feb 2016, T. MacDonald

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from OrderedDict import OrderedDict


# ----------------------------------------------------------------------
#   Indexable Dictionary
# ----------------------------------------------------------------------

class IndexableDict(OrderedDict):
    """ An OrderedDict with list-style access 
    """
        
    def __getitem__(self,k):
        if not isinstance(k,int):
            return super(IndexableDict,self).__getitem__(k)
        else:
            return super(IndexableDict,self).__getitem__(self.keys()[k])

    # iterate on values, not keys
    def __iter__(self):
        return super(IndexableDict,self).itervalues()