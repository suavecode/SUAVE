# OrderedBunch.py
#
# Created:  Aug 2015, T. Lukacyzk
# Modified: Feb 2016, T. MacDonald

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from collections import OrderedDict
from Property import Property
    
# ----------------------------------------------------------------------
#   Ordered Bunch
# ----------------------------------------------------------------------

class OrderedBunch(OrderedDict):
    """ An ordered dictionary that provides attribute-style access.
    """
    
    _root = Property('_root')
    _map  = Property('_map')

    def __delattr__(self, key):
        """od.__delitem__(y) <==> del od[y]"""
        # Deleting an existing item uses self._map to find the link which is
        # then removed by updating the links in the predecessor and successor nodes.
        OrderedDict.__delattr__(self,key)
        link_prev, link_next, key = self._map.pop(key)
        link_prev[1] = link_next
        link_next[0] = link_prev
        
    def __eq__(self, other):
        '''od.__eq__(y) <==> od==y.  Comparison to another OD is order-sensitive
        while comparison to a regular mapping is order-insensitive.

        '''
        if isinstance(other, (OrderedBunch,OrderedDict)):
            return len(self)==len(other) and self.items() == other.items()
        return dict.__eq__(self, other)
        
    def __len__(self):
        return self.__dict__.__len__()   
    
    def __init__(self, items=None, **kwds):
        '''Initialize an ordered dictionary.  Signature is the same as for
           regular dictionaries, but keyword arguments are not recommended
           because their insertion order is arbitrary.
        '''
        
        # od.update(E, **F) -> None.  Update od from dict/iterable E and F.
        # If E is a dict instance, does:           for k in E: od[k] = E[k]
        # If E has a .keys() method, does:         for k in E.keys(): od[k] = E[k]
        # Or if E is an iterable of items, does:   for k, v in E: od[k] = v
        # In either case, this is followed by:     for k, v in F.items(): od[k] = v        
        
            
        def append_value(key,value):               
            self[key] = value            
        
        # a dictionary
        if hasattr(items, 'iterkeys'):
            for key in items.iterkeys():
                append_value(key,items[key])

        elif hasattr(items, 'keys'):
            for key in items.keys():
                append_value(key,items[key])
                
        # items lists
        elif items:
            for key, value in items:
                append_value(key,value)
                
        # key words
        for key, value in kwds.iteritems():
            append_value(key,value) 
            

    def __iter__(self):
        """od.__iter__() <==> iter(od)"""
        root = self._root
        curr = root[1]
        while curr is not root:
            yield curr[2]
            curr = curr[1]
    
    def __new__(klass,*args,**kwarg):

        self = OrderedDict.__new__(klass)
        
        if hasattr(self,'_root'):
            self._root
        else:
            root = [] # sentinel node
            root[:] = [root, root, None]
            dict.__setitem__(self,'_root',root)
            dict.__setitem__(self,'_map' ,{})
        
        return self

    def __reduce__(self):
        """Return state information for pickling"""
        items = [( k, OrderedBunch.__getitem__(self,k) ) for k in OrderedBunch.iterkeys(self)]
        inst_dict = vars(self).copy()
        for k in vars(OrderedBunch()):
            inst_dict.pop(k, None)
        return (_reconstructor, (self.__class__,items,), inst_dict)
    

    def __setattr__(self, key, value):
        """od.__setitem__(i, y) <==> od[i]=y"""
        # Setting a new item creates a new link which goes at the end of the linked
        # list, and the inherited dictionary is updated with the new key/value pair.
        if not hasattr(self,key) and not hasattr(self.__class__,key):
        #if not self.has_key(key) and not hasattr(self.__class__,key):
            root = dict.__getitem__(self,'_root')
            last = root[0]
            map  = dict.__getitem__(self,'_map')
            last[1] = root[0] = map[key] = [last, root, key]
        OrderedDict.__setattr__(self,key, value)


    def __setitem__(self,k,v):
        self.__setattr__(k,v)
        
    def __str__(self,indent=''):
        """ String-form of a Dict.
        """
        
        new_indent = '  '
        args = ''
        
        # trunk data name
        if indent: args += '\n'
        
        # print values   
        for key,value in self.iteritems():
            
            # skip 'hidden' items
            if isinstance(key,str) and key.startswith('_'):
                continue
            
            # recurse into other dict types
            if isinstance(value,OrderedDict):
                if not value:
                    val = '\n'
                else:
                    try:
                        val = value.__str__(indent+new_indent)
                    except RuntimeError: # recursion limit
                        val = ''
                        
            # everything else
            else:
                val = str(value) + '\n'
                
            # this key-value, indented
            args+= indent + str(key) + ' : ' + val
            
        return args     

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
        self.__dict__.clear()
        
    def get(self,k,d=None):
        return self.__dict__.get(k,d)
        
    def has_key(self,k):
        return self.__dict__.has_key(k)


    # allow override of iterators
    __iter = __iter__
    __getitem__ = OrderedDict.__getattribute__
    
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
    
    