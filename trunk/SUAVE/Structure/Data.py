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

