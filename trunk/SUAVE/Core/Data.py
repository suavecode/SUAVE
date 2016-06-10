# Data.py
#
# Created:  Jan 2015, T. Lukacyzk
# Modified: Feb 2016, T. MacDonald

""" SUAVE Data Base Classes
"""

#from DataBunch import DataBunch
from OrderedBunch import OrderedBunch


# for enforcing attribute style access names
import string
chars = string.punctuation + string.whitespace
t_table = string.maketrans( chars          + string.uppercase , 
                            '_'*len(chars) + string.lowercase )

from warnings import warn

# ----------------------------------------------------------------------
#   Data
# ----------------------------------------------------------------------        

class Data(OrderedBunch):
    
    def append(self,value,key=None):
        if key is None: key = value.tag
        key_in = key
        key = key.translate(t_table)
        if key != key_in: warn("changing appended key '%s' to '%s'\n" % (key_in,key))
        if key is None: key = value.tag
        if key in self: raise KeyError, 'key "%s" already exists' % key
        self[key] = value    

    
    def __defaults__(self):
        pass
    
    def __getitem__(self,k):
        if not isinstance(k,int):
            return super(Data,self).__getitem__(k)
        else:
            return super(Data,self).__getitem__(self.keys()[k])
    
    
    def __new__(cls,*args,**kwarg):
        """ supress use of args or kwarg for defaulting
        """
        
        # initialize data, no inputs
        self = super(Data,cls).__new__(cls)
        super(Data,self).__init__()
        
        # get base class list
        klasses = self.get_bases()
                
        # fill in defaults trunk to leaf
        for klass in klasses[::-1]:
            klass.__defaults__(self)
            
        return self
    
    def __init__(self,*args,**kwarg):
        """ initializes DataBunch()
        """
        
        # handle input data (ala class factory)
        input_data = Data.__base__(*args,**kwarg)
        
        # update this data with inputs
        self.update(input_data)
        
    # iterate on values, not keys
    def __iter__(self):
        return super(Data,self).itervalues()
            
    def __str__(self,indent=''):
        new_indent = '  '
        args = ''
        
        # trunk data name
        if not indent:
            args += self.dataname()  + '\n'
        else:
            args += ''
            
        args += super(Data,self).__str__(indent)
        
        return args
        
    def __repr__(self):
        return self.dataname()
    
    #def append(self,value,key=None):
        #if key is None: key = value.tag
        #if key in self: raise KeyError, 'key "%s" already exists' % key
        #self[key] = value    
    
    def find_instances(self,data_type):
        """ DataBunch.find_instances(data_type)
            
            searches DataBunch() for instances of given data_type
            
            Inputs:
                data_type  - a class type, for example type(myData)
                
            Outputs:
                data - DataBunch() of the discovered data
        """
        
        output = Data()
        for key,value in self.iteritems():
            if isinstance(value,type):
                output[key] = value
        return output
    
    def get_bases(self):
        """ find all DataBunch() base classes, return in a list """
        klass = self.__class__
        klasses = []
        while klass:
            if issubclass(klass,Data): 
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
    
