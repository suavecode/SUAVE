
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import re
from Dict import Dict

try:
    from thread import get_ident as _get_ident
except ImportError:
    from dummy_thread import get_ident as _get_ident

try:
    from _abcoll import KeysView, ValuesView, ItemsView
except ImportError:
    pass


# ----------------------------------------------------------------------
#   Ordered Dictionary
# ----------------------------------------------------------------------

class OrderedDict(Dict):
    """Dictionary that remembers insertion order"""
    # The inherited Dict maps keys to values.
    # The inherited Dict provides __getitem__, __len__, __contains__, and get.
    # The remaining methods are order-aware.
    # Big-O running times for all methods are the same as for regular dictionaries.

    # The internal self.__map dictionary maps keys to links in a doubly linked list.
    # The circular doubly linked list starts and ends with a sentinel element.
    # The sentinel element never gets deleted (this simplifies the algorithm).
    # Each link is stored as a list of length three:  [PREV, NEXT, KEY].
    
    __root = None
    __map  = None
    
    def __new__(klass,*args,**kwarg):

        self = super(OrderedDict,klass).__new__(klass)
        
        #if len(args) > 1:
            #raise TypeError('expected at most 1 arguments, got %d' % len(args))
            
        if self.__root is None:
            root = [] # sentinel node
            root[:] = [root, root, None]
            self.__root = root
            self.__map = {}
        
        return self

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
        
        ## result data structure
        #klass = self.__class__
        #from VyPy.data import DataBunch
        #if isinstance(klass,DataBunch):
            #klass = DataBunch
            
        def append_value(key,value):
            #if isinstance(value,dict):
                #value = klass(value)                
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

    def __setitem__(self, key, value):
        """od.__setitem__(i, y) <==> od[i]=y"""
        # Setting a new item creates a new link which goes at the end of the linked
        # list, and the inherited dictionary is updated with the new key/value pair.
        if key not in self:
            root = self.__root
            last = root[0]
            last[1] = root[0] = self.__map[key] = [last, root, key]
        super(OrderedDict,self).__setitem__(key, value)

    def __delitem__(self, key):
        """od.__delitem__(y) <==> del od[y]"""
        # Deleting an existing item uses self.__map to find the link which is
        # then removed by updating the links in the predecessor and successor nodes.
        super(OrderedDict,self).__delitem__(key)
        link_prev, link_next, key = self.__map.pop(key)
        link_prev[1] = link_next
        link_next[0] = link_prev

    def __iter__(self):
        """od.__iter__() <==> iter(od)"""
        root = self.__root
        curr = root[1]
        while curr is not root:
            yield curr[2]
            curr = curr[1]

    def __reversed__(self):
        """od.__reversed__() <==> reversed(od)"""
        root = self.__root
        curr = root[0]
        while curr is not root:
            yield curr[2]
            curr = curr[0]

    def clear(self):
        """od.clear() -> None.  Remove all items from od."""
        try:
            for node in self.__map.itervalues():
                del node[:]
            root = self.__root
            root[:] = [root, root, None]
            self.__map.clear()
        except AttributeError:
            pass
        super(OrderedDict,self).clear()

    def popitem(self, last=True):
        '''od.popitem() -> (k, v), return and remove a (key, value) pair.
           Pairs are returned in LIFO order if last is true or FIFO order if false.
        '''
        if not self:
            raise KeyError('dictionary is empty')
        root = self.__root
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
        del self.__map[key]
        value = super(OrderedDict,self).pop(key)
        return key, value

    # -- the following methods do not depend on the internal structure --
    
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

    def setdefault(self, key, default=None):
        """od.setdefault(k[,d]) -> od.get(k,d), also set od[k]=d if k not in od"""
        if key in self:
            return self[key]
        self[key] = default
        return default

    def __repr__(self, _repr_running={}):
        """od.__repr__() <==> repr(od)"""
        call_key = id(self), _get_ident()
        if call_key in _repr_running:
            return '...'
        _repr_running[call_key] = 1
        try:
            if not self:
                return '%s()' % (self.__class__.__name__,)
            return '%s(%r)' % (self.__class__.__name__, self.items())
        finally:
            del _repr_running[call_key]

    def __reduce__(self):
        """Return state information for pickling"""
        items = [( k, OrderedDict.__getitem__(self,k) ) for k in OrderedDict.iterkeys(self)]
        inst_dict = vars(self).copy()
        for k in vars(OrderedDict()):
            inst_dict.pop(k, None)
        return (_reconstructor, (self.__class__,items,), inst_dict)

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
        if isinstance(other, OrderedDict):
            return len(self)==len(other) and self.items() == other.items()
        return dict.__eq__(self, other)

    def __ne__(self, other):
        return not self == other

    # -- the following methods are only used in Python 2.7 --

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

    def viewkeys(self):
        """od.viewkeys() -> a set-like object providing a view on od's keys"""
        return KeysView(self)

    def viewvalues(self):
        """od.viewvalues() -> an object providing a view on od's values"""
        return ValuesView(self)

    def viewitems(self):
        """od.viewitems() -> a set-like object providing a view on od's items"""
        return ItemsView(self)
    

    def pack_array(self,output='vector'):
        """ OrderedDict.pack_array(output='vector')
            maps the data dict to a 1D vector or 2D column array
            
            Inputs - 
                output - either 'vector' (default), or 'array'
                         chooses whether to output a 1d vector or 
                         2d column array
            Outputs - 
                array - the packed array
                
            Assumptions - 
                will only pack int, float, np.array and np.matrix (max rank 2)
                if using output = 'matrix', all data values must have 
                same length (if 1D) or number of rows (if 2D), otherwise is skipped
        
        """
        
        # dont require dict to have numpy
        import numpy as np
        from VyPy.tools.arrays import atleast_2d_col, array_type, matrix_type
        
        # check output type
        if not output in ('vector','array'): raise Exception , 'output type must be "vector" or "array"'        
        vector = output == 'vector'
        
        # list to pre-dump array elements
        M = []
        
        # valid types for output
        valid_types = ( int, float,
                        array_type,
                        matrix_type )
        
        # initialize array row size (for array output)
        size = [False]
        
        # the packing function
        def do_pack(D):
            for v in D.itervalues():
                # type checking
                if isinstance( v, OrderedDict ): 
                    do_pack(v) # recursion!
                    continue
                elif not isinstance( v, valid_types ): continue
                elif np.rank(v) > 2: continue
                # make column vectors
                v = atleast_2d_col(v)
                # handle output type
                if vector:
                    # unravel into 1d vector
                    v = v.ravel(order='F')
                else:
                    # check array size
                    size[0] = size[0] or v.shape[0] # updates size once on first array
                    if v.shape[0] != size[0]: 
                        #warn ('array size mismatch, skipping. all values in data must have same number of rows for array packing',RuntimeWarning)
                        continue
                # dump to list
                M.append(v)
            #: for each value
        #: def do_pack()
        
        # do the packing
        do_pack(self)
        
        # pack into final array
        if M:
            M = np.hstack(M)
        else:
            # empty result
            if vector:
                M = np.array([])
            else:
                M = np.array([[]])
        
        # done!
        return M
    
    def unpack_array(self,M):
        """ OrderedDict.unpack_array(array)
            unpacks an input 1d vector or 2d column array into the data dictionary
                following the same order that it was unpacked
            important that the structure of the data dictionary, and the shapes
                of the contained values are the same as the data from which the 
                array was packed
        
            Inputs:
                 array - either a 1D vector or 2D column array
                 
            Outputs:
                 a reference to self, updates self in place
                 
        """
        
        # dont require dict to have numpy
        import numpy as np
        from VyPy.tools.arrays import atleast_2d_col, array_type, matrix_type
        
        # check input type
        vector = np.rank(M) == 1
        
        # valid types for output
        valid_types = ( int, float,
                        array_type,
                        matrix_type )
        
        # counter for unpacking
        _index = [0]
        
        # the unpacking function
        def do_unpack(D):
            for k,v in D.iteritems():
                
                # type checking
                if isinstance(v,OrderedDict): 
                    do_unpack(v) # recursion!
                    continue
                elif not isinstance(v,valid_types): continue
                
                # get this value's rank
                rank = np.rank(v)
                
                # get unpack index
                index = _index[0]                
                
                # skip if too big
                if rank > 2: 
                    continue
                
                # scalars
                elif rank == 0:
                    if vector:
                        D[k] = M[index]
                        index += 1
                    else:#array
                        continue
                        #raise RuntimeError , 'array size mismatch, all values in data must have same number of rows for array unpacking'
                    
                # 1d vectors
                elif rank == 1:
                    n = len(v)
                    if vector:
                        D[k][:] = M[index:(index+n)]
                        index += n
                    else:#array
                        D[k][:] = M[:,index]
                        index += 1
                    
                # 2d arrays
                elif rank == 2:
                    n,m = v.shape
                    if vector:
                        D[k][:,:] = np.reshape( M[index:(index+(n*m))] ,[n,m], order='F')
                        index += n*m 
                    else:#array
                        D[k][:,:] = M[:,index:(index+m)]
                        index += m
                
                #: switch rank
                
                _index[0] = index

            #: for each itme
        #: def do_unpack()
        
        # do the unpack
        do_unpack(self)
         
        # check
        if not M.shape[-1] == _index[0]: warn('did not unpack all values',RuntimeWarning)
         
        # done!
        return self
    
    def do_recursive(self,method,other=None,default=None):
        
        # result data structure
        klass = self.__class__
        from VyPy.data import DataBunch
        if isinstance(klass,DataBunch):
            klass = DataBunch
        result = klass()
                
        # the update function
        def do_operation(A,B,C):
            for k,a in A.iteritems():
                if isinstance(B,OrderedDict):
                    if B.has_key(k):
                        b = B[k]
                    else: 
                        continue
                else:
                    b = B
                # recursion
                if isinstance(a,OrderedDict):
                    c = klass()
                    C[k] = c
                    do_operation(a,b,c)
                # method
                else:
                    if b is None:
                        c = method(a)
                    else:
                        c = method(a,b)
                    if not c is None:
                        C[k] = c
                #: if type
            #: for each key,value
        #: def do_operation()
        
        # do the update!
        do_operation(self,other,result)    
        
        return result
        


# for rebuilding dictionaries with attributes
def _reconstructor(klass,items):
    self = OrderedDict.__new__(klass)
    OrderedDict.__init__(self,items)
    return self



# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------

if __name__ == '__main__':
    
    # pack it up
    o = OrderedDict()
    o['x'] = 'hello'
    o['y'] = 1
    o['z'] = [3,4,5]
    o['t'] = OrderedDict()
    o['t']['h'] = 20
    o['t']['i'] = (1,2,3)

    # printing
    print o

    # pickling
    import pickle

    d = pickle.dumps(o)
    p = pickle.loads(d)
    
    print ''
    print p    
    
    # recursive updates
    o['t']['h'] = 'changed'
    p.update(o)

    print ''
    print p
    
    import numpy as np
    import VyPy
    a = OrderedDict()
    a['f'] = 1
    a['g'] = 2
    a['b'] = OrderedDict()
    a['b']['h'] = np.array([1,2,3])
    
    from copy import deepcopy
    b = deepcopy(a)
    
    def method(self,other):
        return self-other
    
    c = a.do_recursive(method,b)
    
    print c
    
    def method(self):
        if isinstance(self,np.ndarray):
            return self[-1]
        else:
            return None
    
    d = a.do_recursive(method)
    print d
    
    wait = 0
    

    
    
    
# ----------------------------------------------------------------------
#   Gravetart
# ----------------------------------------------------------------------

    #class TestDescriptor(object):
        #def __init__(self,x):
            #self.x = x
        
        #def __get__(self,obj,kls=None):
            #print '__get__'
            #print type(obj), type(self)
            #print self.x
            #return self.x
        
        #def __set__(self,obj,val):
            #print '__set__'
            #print type(obj), type(self)
            #print val
            #self.x = val
        
    #class TestObject(OrderedDict):
        #pass
        ##def __init__(self,c):
            ##self.c = c
    
    #o = TestObject()
    #o['x'] = TestDescriptor([1,2,3])
    #o['y'] = TestDescriptor([4,3,5])
    #for i in range(10):
        #o['x%i'%i] = 'yo'    
    
    #print ''
    #print o
    ##print o.c
    
    #print ''
    #o['x'] = [3,4,5]
            
    #print ''
    #print 'pickle'
    #import pickle
        
    #d = pickle.dumps(o)
    #p = pickle.loads(d)
    
    #print ''
    #print p
    ##print p.c