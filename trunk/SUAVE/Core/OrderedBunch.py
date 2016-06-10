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
    
    
    # prettier printing
    def __repr__(self):
        """ Invertible* string-form of a Dict.
        """
        keys = self.keys()
        args = ', '.join(['%s=%r' % (key, self[key]) for key in keys if not key.startswith('_')])
        return '%s(%s)' % (self.__class__.__name__, args)
    

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

    def to_dict(self):
        r = {}
        for k,v in self.items():
            if isinstance(v,OrderedDict):
                v = v.to_dict()
            r[k] = v
        return r
        
    def do_recursive(self,method,other=None,default=None):
        
        # result data structure
        klass = self.__class__
        from DataBunch import DataBunch
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
        
        # do the update!
        do_operation(self,other,result)    
        
        return result        

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
        from Arrays import atleast_2d_col, array_type, matrix_type
        
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
        from Arrays import atleast_2d_col, array_type, matrix_type
        
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
        
        # do the unpack
        do_unpack(self)
         
        # check
        if not M.shape[-1] == _index[0]: warn('did not unpack all values',RuntimeWarning)
         
        # done!
        return self    
    
    
    def update(self,other):
        """ Dict.update(other)
            updates the dictionary in place, recursing into additional
            dictionaries inside of other
            
            Assumptions:
              skips keys that start with '_'
        """
        if not isinstance(other,dict):
            raise TypeError , 'input is not a dictionary type'
        for k,v in other.iteritems():
            # recurse only if self's value is a Dict()
            if k.startswith('_'):
                continue
        
            try:
                self[k].update(v)
            except:
                self[k] = v
        return 


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
    
    