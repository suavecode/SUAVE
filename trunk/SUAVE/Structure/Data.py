# Data.py
#

""" SUAVE Data Base Classes
"""

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import types

from Indexable_Bunch import Indexable_Bunch 
from Data_Exception  import Data_Exception
from Data_Warning    import Data_Warning
from copy            import deepcopy
from warnings        import warn
from collections     import OrderedDict


# ----------------------------------------------------------------------
#   Data Base Class
# ----------------------------------------------------------------------        
        
class Data(Indexable_Bunch):
    """ SUAVE.Structure.Data()
        
        a dict-type container with attribute, item and index style access
        initializes with default attributes
        will recursively search for defaults of base classes
        current class defaults overide base class defaults

        Methods:
            __defaults__(self)      : sets the defaults of 
            find_instances(datatype)
    """
    
    def __defaults__(self):
        pass
    
    def __new__(cls,*args,**kwarg):
        """ supress use of args or kwarg for defaulting
        """
        
        # initialize data, no inputs
        self = Indexable_Bunch.__new__(cls)
        Indexable_Bunch.__init__(self)
        
        # get base class list
        klasses = self.get_bases()
                
        # fill in defaults trunk to leaf
        for klass in klasses[::-1]:
            klass.__defaults__(self)
        
        ## ensure local copies
        #for k,v in self.iteritems():
            #self[k] = deepcopy(v)
            
        return self

    def __init__(self,*args,**kwarg):
        """ initializes Data()
        """
        
        # handle input data (ala class factory)
        input_data = Indexable_Bunch(*args,**kwarg)
        
        # update this data with inputs
        self.update(input_data)
        
        # call over-ridable post-initialition setup
        self.__check__()
        
    #: def __init__()
    
    def __check__(self):
        """ 
        """
        pass
    
    def __setitem__(self,k,v):
        # attach all functions as static methods
        if isinstance(v,types.FunctionType):
            v = staticmethod(v)
        Indexable_Bunch.__setitem__(self,k,v)
        
    def __str__(self,indent=''):
        new_indent = '  '
        args = ''
        
        # trunk data name
        if not indent:
            args += self.dataname()  + '\n'
        else:
            args += '\n'
         
        # print values   
        for key,value in self.iteritems():
            if isinstance(value,Data):
                if not value:
                    val = '()\n'
                else:
                    try:
                        val = value.__str__(indent+new_indent)
                    except RuntimeError: # recursion limit
                        val = ''
            else:
                val = str(value) + '\n'
            args+= indent + str(key) + ' = ' + val
            
        return args
            
    def __repr__(self):
        return self.__str__()
       
    def __reduce__(self):
        t = Indexable_Bunch.__reduce__(self)
        cls   = t[0]
        items = t[1]
        items = (cls,items)
        args  = t[2:]
        return (DataReConstructor,items) + args
    
    #def __iter__(self):
        #for k in super(Data,self).__iter__():
            #yield (k, self[k])
        
    def find_instances(self,data_type):
        """ SUAVE.Data.find_instances(data_type)
            
            searches Data() for instances of given data_type
            
            Inputs:
                data_type  - a class type, for example type(myData)
                
            Outputs:
                data - Data() dictionary of the discovered data
        """
        
        output = Data()
        for key,value in self.iteritems():
            if isinstance(value,type):
                output[key] = value
        return output
    
    def pack_array(self,output='vector'):
        """ Data.pack_array(output='vector')
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
        
        # dont require data to have numpy
        import numpy as np
        from SUAVE.Methods.Utilities import atleast_2d_col
        
        # check output type
        if not output in ('vector','array'): raise Exception , 'output type must be "vector" or "array"'        
        vector = output == 'vector'
        
        # list to pre-dump array elements
        M = []
        
        # valid types for output
        valid_types = ( int, float,
                        np.ndarray,
                        np.matrixlib.defmatrix.matrix )
        
        # initialize array row size (for array output)
        size = [False]
        
        # the packing function
        def do_pack(D):
            for v in D.itervalues():
                # type checking
                if isinstance( v, Data ): 
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
            if vector:
                M = np.array([])
            else:
                M = np.array([[]])
        
        # done!
        return M
    
    def unpack_array(self,M):
        """ Data.unpack_array(array)
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
        
        # dont require data to have numpy
        import numpy as np
        from SUAVE.Methods.Utilities import atleast_2d_col
        
        # check input type
        vector = np.rank(M) == 1
        
        # valid types for output
        valid_types = ( int, float,
                        np.ndarray,
                        np.matrixlib.defmatrix.matrix )
        
        # counter for unpacking
        _index = [0]
        
        # the unpacking function
        def do_unpack(D):
            for k,v in D.iteritems():
                
                # type checking
                if isinstance(v,Data): 
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
    
    def get_bases(self):
        """ find all Data() base classes, return in a list """
        klass = self.__class__
        klasses = []
        while klass:
            if issubclass(klass,Data): 
                klasses.append(klass)
                klass = klass.__base__
            else:
                klass = None
        if not klasses: # empty list
            raise Data_Exception , 'class %s is not of type Data()' % self.__class__
        return klasses
    
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

    def linked_copy(self,key=None):
        """ SAUVE.Data.linked_copy(key=None)
            returns a copy of the Data dictionary
            the copy's values are referenced to the original's values
            value changes to the original will propogate to the copy
            value changes to the copy will break the link to the copy
            new values added to the copy will *not*  propogate to the copy
            copied values that reference deleted original values will return a BrokenKey() object
        """
        if not key is None:
            return LinkedValue(self,key)
        
        # else ...
        
        kopy = DataReConstructor(self.__class__)
        for key,value in self.iteritems():
            if isinstance(value,(Data,LinkedValue)):
                kopy[key] = value.linked_copy()
            else:
                kopy[key] = LinkedValue(self,key)
        return kopy
    
    def is_link(self,key):
        """ returns True if the key's value is LinkedCopy
        """
        value = OrderedDict.__getitem__(self,key)
        return isinstance(value,LinkedValue)

#: class Data()


# ----------------------------------------------------------------------
#   Linked Value
# ----------------------------------------------------------------------        

class LinkedValue(object):  
    
    def __init__(self,data,key):
        self._data = data
        self._key  = key
    
    def __get__(self,obj,typ=None):
        try:
            return self._data[self._key]
        except:
            return BrokenLink()
        
    def linked_copy(self):
        return LinkedValue(self._data,self._key)

#: class LinkedValue()


# ----------------------------------------------------------------------
#   Broken Link
# ----------------------------------------------------------------------        

class BrokenLink(object):

    def __str__(self):

        return '<Broken Link>'
    def __repr__(self):
        return self.__str__()

    def __nonzero__(self):
        return False    

#: BrokenKey()


# ----------------------------------------------------------------------
#   Data Reconstructor
# ----------------------------------------------------------------------        

def DataReConstructor(klass,items=(),**kwarg):
    """ reconstructs a Data()-type instance from pickle or deepcopy
    """
    # initialize data, no inputs
    self = Indexable_Bunch.__new__(klass)
    Indexable_Bunch.__init__(self,*items,**kwarg)
    return self

#: def DataConstructor()


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
    m.hieght = ones * 1.
    m.rates = Data()
    m.rates.angle  = ones * 3.14
    m.rates.slope  = ones * 20.
    m.rates.special = 'nope'
    m.value = 1.0
    
    print m
    
    V = m.pack_array('vector')
    M = m.pack_array('array')
    
    print V
    print M
    
    V = V*10
    M = M-10
    
    print m.unpack_array(V)
    print m.unpack_array(M)
    
