#!/usr/bin/env python

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from Bunch import Bunch
from OrderedDict import OrderedDict
from Property import Property
    
# ----------------------------------------------------------------------
#   Ordered Bunch
# ----------------------------------------------------------------------

class OrderedBunch(Bunch,OrderedDict):
    """ An ordered dictionary that provides attribute-style access.
    """
    
    _root = Property('_root')
    _map  = Property('_map')
    
    def __new__(klass,*args,**kwarg):

        self = Bunch.__new__(klass)
        
        if len(args) > 1:
            raise TypeError('expected at most 1 arguments, got %d' % len(args))
        try:
            self._root
        except:
            root = [] # sentinel node
            root[:] = [root, root, None]
            dict.__setitem__(self,'_root',root)
            dict.__setitem__(self,'_map' ,{})
        
        return self

    def __setattr__(self, key, value):
        """od.__setitem__(i, y) <==> od[i]=y"""
        # Setting a new item creates a new link which goes at the end of the linked
        # list, and the inherited dictionary is updated with the new key/value pair.
        if not hasattr(self,key) and not hasattr(self.__class__,key):
            root = dict.__getitem__(self,'_root')
            last = root[0]
            map  = dict.__getitem__(self,'_map')
            last[1] = root[0] = map[key] = [last, root, key]
        Bunch.__setattr__(self,key, value)

    def __delattr__(self, key):
        """od.__delitem__(y) <==> del od[y]"""
        # Deleting an existing item uses self._map to find the link which is
        # then removed by updating the links in the predecessor and successor nodes.
        Bunch.__delattr__(self,key)
        link_prev, link_next, key = self._map.pop(key)
        link_prev[1] = link_next
        link_next[0] = link_prev
        
    def __setitem__(self,k,v):
        self.__setattr__(k,v)
    
    def __delitem__(self,k):
        self.__delattr__(k)

    def __iter__(self):
        """od.__iter__() <==> iter(od)"""
        root = self._root
        curr = root[1]
        while curr is not root:
            yield curr[2]
            curr = curr[1]

    def __reversed__(self):
        """od.__reversed__() <==> reversed(od)"""
        root = self._root
        curr = root[0]
        while curr is not root:
            yield curr[2]
            curr = curr[0]

    def clear(self):
        """od.clear() -> None.  Remove all items from od."""
        try:
            for node in self._map.itervalues():
                del node[:]
            root = self._root
            root[:] = [root, root, None]
            self._map.clear()
        except AttributeError:
            pass
        Bunch.clear(self)

    def popitem(self, last=True):
        '''od.popitem() -> (k, v), return and remove a (key, value) pair.
           Pairs are returned in LIFO order if last is true or FIFO order if false.
        '''
        if not self:
            raise KeyError('dictionary is empty')
        root = self._root
        if last:
            link = root[0]
            link_prev = link[0]
            link_prev[1] = root
            root[0] = link_prev
        else:
            link = root[1]
            link_next = link[1]
            root[1] = link_next
            link_next[0] = root
        key = link[2]
        del self._map[key]
        value = Bunch.pop(self,key)
        return key, value
        
    def __reduce__(self):
        """Return state information for pickling"""
        items = [( k, OrderedBunch.__getitem__(self,k) ) for k in OrderedBunch.iterkeys(self)]
        inst_dict = vars(self).copy()
        for k in vars(OrderedBunch()):
            inst_dict.pop(k, None)
        return (_reconstructor, (self.__class__,items,), inst_dict)
    
    
    __marker = object()

    def pop(self, key, default=__marker):
        '''od.pop(k[,d]) -> v, remove specified key and return the corresponding value.
           If key is not found, d is returned if given, otherwise KeyError is raised.
        '''
        if key in self:
            result = self[key]
            del self[key]
            return result
        if default is self.__marker:
            raise KeyError(key)
        return default

    def copy(self):
        """od.copy() -> a shallow copy of od"""
        return self.__class__(self)

    @classmethod
    def fromkeys(cls, iterable, value=None):
        '''OD.fromkeys(S[, v]) -> New ordered dictionary with keys from S
        and values equal to v (which defaults to None).

        '''
        d = cls()
        for key in iterable:
            d[key] = value
        return d

    def __eq__(self, other):
        '''od.__eq__(y) <==> od==y.  Comparison to another OD is order-sensitive
        while comparison to a regular mapping is order-insensitive.

        '''
        if isinstance(other, (OrderedBunch,OrderedDict)):
            return len(self)==len(other) and self.items() == other.items()
        return dict.__eq__(self, other)

    def __ne__(self, other):
        return not self == other

    # allow override of iterators
    __iter = __iter__
    
    def keys(self):
        """OrderedDict.keys() -> list of keys in the dictionary"""
        return list(self.__iter())
    
    def values(self):
        """OrderedDict.values() -> list of values in the dictionary"""
        return [self[key] for key in self.__iter()]
    
    def items(self):
        """OrderedDict.items() -> list of (key, value) pairs in the dictionary"""
        return [(key, self[key]) for key in self.__iter()]
    
    def iterkeys(self):
        """OrderedDict.iterkeys() -> an iterator over the keys in the dictionary"""
        return self.__iter()
    
    def itervalues(self):
        """OrderedDict.itervalues -> an iterator over the values in the dictionary"""
        for k in self.__iter():
            yield self[k]
    
    def iteritems(self):
        """od.iteritems -> an iterator over the (key, value) items in the dictionary"""
        for k in self.__iter():
            yield (k, self[k])    

    
# for rebuilding dictionaries with attributes
def _reconstructor(klass,items):
    self = OrderedBunch.__new__(klass)
    OrderedBunch.__init__(self,items)
    return self
        
        

# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------

if __name__ == '__main__':
    
    o = OrderedBunch()
    print 'should be zero:' , len(o)
    
    o['x'] = 'hello'
    o.y = 1
    o['z'] = [3,4,5]
    o.t = OrderedBunch()
    o.t['h'] = 20
    o.t.i = (1,2,3)
    
    print o

    import pickle

    d = pickle.dumps(o)
    p = pickle.loads(d)
    
    print ''
    print p    

    o.t['h'] = 'changed'
    p.update(o)
    
    print ''
    print p
    
    class TestClass(OrderedBunch):
        a = Property('a')
        def __init__(self):
            self.a = 'hidden!'
            self.b = 'hello!'
    
    c = TestClass()
    print c
    
    