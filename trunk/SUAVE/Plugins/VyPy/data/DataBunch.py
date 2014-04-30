
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from IndexableBunch import IndexableBunch ## Too Slow on __getitem__
from OrderedBunch import OrderedBunch 

import types
from copy            import deepcopy
from warnings        import warn

import numpy as np
from VyPy.tools.arrays import atleast_2d_col, array_type, matrix_type


# ----------------------------------------------------------------------
#   Data Dictionary
# ----------------------------------------------------------------------

class DataBunch(IndexableBunch):
    """ DataBunch()
        
        a dict-type container with attribute and item style access
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
        self = super(DataBunch,cls).__new__(cls)
        super(DataBunch,self).__init__()
        
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
        """ initializes DataBunch()
        """
        
        # handle input data (ala class factory)
        input_data = DataBunch.__base__(*args,**kwarg)
        
        # update this data with inputs
        self.update(input_data)
        
        # call over-ridable post-initialition setup
        self.__check__()
        
    #: def __init__()
    
    def __check__(self):
        """ 
        """
        pass
            
    def __str__(self,indent=''):
        new_indent = '  '
        args = ''
        
        # trunk data name
        if not indent:
            args += self.dataname()  + '\n'
        else:
            args += ''
            
        args += super(DataBunch,self).__str__(indent)
        
        return args
        
    def __repr__(self):
        return self.__str__()
    
    def append(self,value,key=None):
        if key is None: key = value.tag
        if key in self: raise KeyError, 'key "%s" already exists' % key
        self[key] = value    
    
    def find_instances(self,data_type):
        """ DataBunch.find_instances(data_type)
            
            searches DataBunch() for instances of given data_type
            
            Inputs:
                data_type  - a class type, for example type(myData)
                
            Outputs:
                data - DataBunch() of the discovered data
        """
        
        output = DataBunch()
        for key,value in self.iteritems():
            if isinstance(value,type):
                output[key] = value
        return output
    
    def get_bases(self):
        """ find all DataBunch() base classes, return in a list """
        klass = self.__class__
        klasses = []
        while klass:
            if issubclass(klass,DataBunch): 
                klasses.append(klass)
                klass = klass.__base__
            else:
                klass = None
        if not klasses: # empty list
            raise TypeError , 'class %s is not of type DataBunch()' % self.__class__
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


    
# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------        

if __name__ == '__main__':
    

    d = DataBunch()
    d.tag = 'data name'
    d['value'] = 132
    d.options = DataBunch()
    d.options.field = 'of greens'
    d.options.half  = 0.5
    print d
    
    
    ones = np.ones([10,1])
        
    m = DataBunch()
    m.tag = 'numerical data'
    m.hieght = ones * 1.
    m.rates = DataBunch()
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
    
    
    # speed test
    from time import time, sleep
    t0 = time()
    for i in range(100000):
        v = d.options.half
    t1 = time()-t0
    
    class SimpleBunch:
        pass
    z = SimpleBunch()
    z.t = SimpleBunch
    z.t.i = 0
    t0 = time()
    for i in range(100000):
        v = z.t.i
    t2 = time()-t0
    
    print 'Bunch:       %.6f' % (t1)
    print 'SimpleBunch: %.6f' % (t2)    
    