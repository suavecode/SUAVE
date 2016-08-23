# Data.py
#
# Created:  Jun 2016, E. Botero
# Modified:

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

# for enforcing attribute style access names
import string
from warnings import warn
chars = string.punctuation + string.whitespace
t_table = string.maketrans( chars          + string.uppercase , 
                            '_'*len(chars) + string.lowercase )

dictgetitem = dict.__getitem__
objgetattrib = object.__getattribute__

# ----------------------------------------------------------------------
#   Data
# ----------------------------------------------------------------------        

class Data(dict):
    """"""
    
    def __getattribute__(self, k):
        try:
            return dictgetitem(self,k)
        except:
            return objgetattrib(self,k)

    def __setattr__(self, k, v):
        try:
            objgetattrib(self, k)
        except:
            self[k] = v
        else:          
            object.__setattr__(self, k, v) 
            
    def __delattr__(self, k):
        try:
            objgetattrib(self, k)
        except:
            del self[k]
        else:
            object.__delattr__(self, k)
    
    def __defaults__(self):
        pass      
    
    def __new__(cls,*args,**kwarg):
        
        # initialize data, no inputs
        self = super(Data,cls).__new__(cls)
        super(Data,self).__init__() 
        
        # get base class list
        klasses = self.get_bases()
                
        # fill in defaults trunk to leaf
        for klass in klasses[::-1]:
            klass.__defaults__(self)
            
        return self
    
    def typestring(self):
        # build typestring
        typestring = str(type(self)).split("'")[1]
        typestring = typestring.split('.')
        if typestring[-1] == typestring[-2]:
            del typestring[-1]
        typestring = '.'.join(typestring) 
        return typestring    
    
    def dataname(self):
        return "<data object '" + self.typestring() + "'>"     
    
    
    def __str__(self,indent=''):
        new_indent = '  '
        args = ''
        
        # trunk data name
        if not indent:
            args += self.dataname()  + '\n'
        else:
            args += ''
            
        args += self.__str2(indent)
        
        return args    
    
    
    def __str2(self,indent=''):
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
            if isinstance(value,dict):
                if not value:
                    val = '\n'
                else:
                    try:
                        val = value.__str2(indent+new_indent)
                    except RuntimeError: # recursion limit
                        val = ''
                        
            # everything else
            else:
                val = str(value) + '\n'
                
            # this key-value, indented
            args+= indent + str(key) + ' : ' + val
            
        return args        
    
    def __init__(self,*args,**kwarg):

        # handle input data (ala class factory)
        input_data = Data.__base__(*args,**kwarg)
        
        # update this data with inputs
        self.update(input_data)    

    def __iter__(self):
        return self.itervalues()
    
    def itervalues(self):
        for k in super(Data,self).__iter__():
            yield self[k]
            
    def values(self):
        return self.__values()          
            
    def __values(self):
        return [self[key] for key in super(Data,self).__iter__()]    
    
    def update(self,other):
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
    
    def get_bases(self):
        klass = self.__class__
        klasses = []
        while klass:
            if issubclass(klass,Data): 
                klasses.append(klass)
                klass = klass.__base__
            else:
                klass = None
        if not klasses: # empty list
            raise TypeError , 'class %s is not of type Data()' % self.__class__
        return klasses    
    
    def append(self,value,key=None):
        if key is None: key = value.tag
        key_in = key
        key = key.translate(t_table)
        if key != key_in: warn("changing appended key '%s' to '%s'\n" % (key_in,key))
        if key is None: key = value.tag
        if key in self: raise KeyError, 'key "%s" already exists' % key
        self[key] = value        
    
    def deep_set(self,keys,val):
        
        if isinstance(keys,str):
            keys = keys.split('.')
        
        data = self
         
        if len(keys) > 1:
            for k in keys[:-1]:
                data = data[k]
        
        data[ keys[-1] ] = val
        
        return data

    def deep_get(self,keys):
        
        if isinstance(keys,str):
            keys = keys.split('.')
        
        data = self
         
        if len(keys) > 1:
            for k in keys[:-1]:
                data = data[k]
        
        value = data[ keys[-1] ]
        
        return value
        
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
                if isinstance( v, dict ): 
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
                if isinstance(v, dict): 
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
    
    def do_recursive(self,method,other=None,default=None):
    
        # result data structure
        klass = self.__class__
        if isinstance(klass,Data):
            klass = Data
        result = klass()
    
        # the update function
        def do_operation(A,B,C):
            for k,a in A.iteritems():
                if isinstance(B,Data):
                    if B.has_key(k):
                        b = B[k]
                    else: 
                        continue
                else:
                    b = B
                # recursion
                if isinstance(a,Data):
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

def bunchify(x):
    if isinstance(x, dict):
        return Bunch( (k, bunchify(v)) for k,v in iteritems(x) )
    elif isinstance(x, (list, tuple)):
        return type(x)( bunchify(v) for v in x )
    else:
        return x

def unbunchify(x):
    if isinstance(x, dict):
        return dict( (k, unbunchify(v)) for k,v in iteritems(x) )
    elif isinstance(x, (list, tuple)):
        return type(x)( unbunchify(v) for v in x )
    else:
        return x
        

# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------        

if __name__ == '__main__':
    
    d = Data()
    d.tag = 'data name'
    d['value'] = 132
    d.options = Data()
    d.options.field = 'of greens'
    d.options.half  = 0.5
    print d
    
    import numpy as np
    ones = np.ones([10,1])
        
    m = Data()
    m.tag = 'numerical data'
    m.height = ones * 1.
    m.rates = Data()
    m.rates.angle  = ones * 3.14
    m.rates.slope  = ones * 20.
    m.rates.special = 'nope'
    m.value = 1.0
    
    print m
    