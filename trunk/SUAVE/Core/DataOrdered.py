## @ingroup Core
# DataOrdered.py
#
# Created:  Jul 2016, E. Botero
# Modified: Sep 2016, E. Botero
#           May 2020, E. Botero
#           Jul 2020, E. Botero 
#           Jul 2021, E. Botero

   
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------  

from collections import OrderedDict

# for enforcing attribute style access names
import string
chars = string.punctuation + string.whitespace
t_table = str.maketrans( chars          + string.ascii_uppercase , 
                            '_'*len(chars) + string.ascii_lowercase )

import numpy as np

# ----------------------------------------------------------------------
#   Property Class
# ----------------------------------------------------------------------   

class Property(object):
    """ Used to create the root map essential to the linking in DataOrdered()
       
        Assumptions:
        N/A
        
        Source:
        N/A
    """    
    
    def __init__(self,key=None):
        """ Initializes a property
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """           
        self._key = key
        
    def __get__(self,obj,kls=None):
        """ Gets a property
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            obj
    
            Outputs:
            self.key
    
            Properties Used:
            N/A    
        """           
        if obj is None: return self
        else          : return dict.__getitem__(obj,self._key)
        
    def __set__(self,obj,val):
        """ Sets a property
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            obj
            value
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """          
        dict.__setitem__(obj,self._key,val)
        
    def __delete__(self,obj):
        """ Deletes a property
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """          
        dict.__delitem__(obj,self._key)

    
# ----------------------------------------------------------------------
#   DataOrdered
# ----------------------------------------------------------------------        

## @ingroup Core
class DataOrdered(OrderedDict):
    """ An extension of the Python dict which allows for both tag and '.' usage.
        This is an ordered dictionary. So indexing it will produce deterministic results.
       
        Assumptions:
        N/A
        
        Source:
        N/A
    """
    
    
    _root = Property('_root')
    _map  = Property('_map')    
    
    def append(self,value,key=None):
        """ Adds new values to the classes. Can also change an already appended key
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            value
            key
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """         
        if key is None: key = value.tag
        key = key.translate(t_table)
        if key is None: key = value.tag
        if key in self: raise KeyError('key "%s" already exists' % key)
        self.__setattr__(key,value)    

    def __defaults__(self):
        """ A stub for all classes that come later
            
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """         
        pass
    
    def __getitem__(self,k):
        """ Retrieves an attribute set by a key k
            
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """          
        if not (isinstance(k,int) or isinstance(k,np.int64)):
            return super(DataOrdered,self).__getattribute__(k)
        else:
            return super(DataOrdered,self).__getattribute__(self.keys()[k])
    
    def __new__(cls,*args,**kwarg):
        """ Creates a new Data() class
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """         
        # Make the new:
        self = OrderedDict.__new__(cls)
        
        if self.hasattr('_root'):
            self._root
        else:
            root = [] # sentinel node
            root[:] = [root, root, None]
            dict.__setitem__(self,'_root',root)
            dict.__setitem__(self,'_map' ,{})        
        
        # Use the base init
        self.__init2()
        
        # get base class list
        klasses = self.get_bases()
                
        # fill in defaults trunk to leaf
        for klass in klasses[::-1]:
            klass.__defaults__(self)
            
        return self
    
    def hasattr(self,k):
        try:
            self.__getitem__(k)
            return True
        except:
            return False
            
    
    def __init__(self,*args,**kwarg):
        """ Initializes a new Data() class
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """         

        # handle input data (ala class factory)
        input_data = DataOrdered.__base__(*args,**kwarg)
        
        # update this data with inputs
        self.update(input_data)
        
        
    def __init2(self, items=None, **kwds):
        """ A helper that allows __init_ to complete the new Data() class
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """         
        def append_value(key,value):  
            
            self[key] = value            
        
        # a dictionary
        if hasattr(items, 'iterkeys'):
            for key in items.keys():
                append_value(key,items[key])

        elif hasattr(items, 'keys'):
            for key in items.keys():
                append_value(key,items[key])
                
        # items lists
        elif items:
            for key, value in items:
                append_value(key,value)
                
        # key words
        for key, value in kwds.items():
            append_value(key,value)     

    # iterate on values, not keys
    def __iter__(self):
        """ Returns all the iterable values. Can be used in a for loop.
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """          
        return iter(self.values())
            
    def __str__(self,indent=''):
        """ This function is used for printing the class. This starts the first line of printing.
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """         
        new_indent = '  '
        args = ''
        
        # trunk data name
        if not indent:
            args += self.dataname()  + '\n'
        else:
            args += ''
            
        args += self.__str2(indent)
        
        return args
    
    def get_bases(self):
        """ Finds the higher classes that may be built off of data
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            klasses
    
            Properties Used:
            N/A    
        """        
        # Get the Method Resolution Order, i.e. the ancestor tree
        klasses = list(self.__class__.__mro__)
        
        # Make sure that this is a Data object, otherwise throw an error.
        if DataOrdered not in klasses:
            raise TypeError('class %s is not of type Data()' % self.__class__)    
        
        # Remove the last two items, dict and object. Since the line before ensures this is a data object this won't break
        klasses = klasses[:-3]

        return klasses 
    
    def typestring(self):
        """ This function makes the .key.key structure in string form of Data()
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """           
        typestring = str(type(self)).split("'")[1]
        typestring = typestring.split('.')
        if typestring[-1] == typestring[-2]:
            del typestring[-1]
        typestring = '.'.join(typestring) 
        return typestring
    
    def dataname(self):
        """ This function is used for printing the class
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """        
        return "<data object '" + self.typestring() + "'>"

    def deep_set(self,keys,val):
        """ Regresses through a list of keys the same value in various places in a dictionary.
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            keys  - The keys to iterate over
            val   - The value to be set
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """         
        
        if isinstance(keys,str):
            keys = keys.split('.')
        
        data = self
         
        if len(keys) > 1:
            for k in keys[:-1]:
                data = data[k]
        
        data[ keys[-1] ] = val
        
        return data

    def deep_get(self,keys):
        """ Regresses through a list of keys to pull a specific value out
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            keys  - The keys to iterate over
            
            Outputs:
            value - The value to be retrieved
    
            Properties Used:
            N/A    
        """          
        
        if isinstance(keys,str):
            keys = keys.split('.')
        
        data = self
         
        if len(keys) > 1:
            for k in keys[:-1]:
                data = data[k]
        
        value = data[ keys[-1] ]
        
        return value   
    
    def update(self,other):
        """ Updates the internal values of a dictionary with given data
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            other
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """          
        if not isinstance(other,dict):
            raise TypeError('input is not a dictionary type')
        for k,v in other.items():
            # recurse only if self's value is a Dict()
            if k.startswith('_'):
                continue
        
            try:
                self[k].update(v)
            except:
                self[k] = v
        return 

    def __delattr__(self, key):
        """ An override of the standard __delattr_ in Python. This deletes whatever is called by k
            
            Assumptions:
            This one tries to treat k as an object, if that fails it treats it as a key.
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """            
        # Deleting an existing item uses self._map to find the link which is
        # then removed by updating the links in the predecessor and successor nodes.
        OrderedDict.__delattr__(self,key)
        link_prev, link_next, key = self._map.pop(key)
        link_prev[1] = link_next
        link_next[0] = link_prev
        
    def __eq__(self, other):
        """ This is overrides the Python function for checking for equality
            
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """           
        if isinstance(other, (DataOrdered,OrderedDict)):
            return len(self)==len(other) and np.all(self.items() == other.items())
        return dict.__eq__(self, other)
        
    def __len__(self):
        """ This is overrides the Python function for checking length
            
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """          
        return self.__dict__.__len__()   

    def __iter_basic__(self):
        """ Returns all the iterable values. Can be used in a for loop.
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """           
        root = self._root
        curr = root[1]
        while curr is not root:
            yield curr[2]
            curr = curr[1]
            
    def __reduce__(self):
        """ Reduction function used for making configs
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """          
        items = [( k, DataOrdered.__getitem2(self,k) ) for k in DataOrdered.iterkeys(self)]
        inst_dict = vars(self).copy()
        for k in vars(DataOrdered()):
            inst_dict.pop(k, None)
        return (_reconstructor, (self.__class__,items,), inst_dict)
    
    def __setattr__(self, key, value):
        """ An override of the standard __setattr_ in Python.
            
            Assumptions:
            This one tries to treat k as an object, if that fails it treats it as a key.
    
            Source:
            N/A
    
            Inputs:
            key        [key]
            value        [value]
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """        
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
        """ An override of the standard __setattr_ in Python.
            
            Assumptions:
            This one tries to treat k as an object, if that fails it treats it as a key.
    
            Source:
            N/A
    
            Inputs:
            k        [key]
            v        [value]
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """        
        self.__setattr__(k,v)
        
    def __str2(self,indent=''):
        """ This regresses through and does the rest of printing that __str__ missed
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """        
        
        new_indent = '  '
        args = ''
        
        # trunk data name
        if indent: args += '\n'
        
        # print values   
        for key,value in self.items():
            
            # skip 'hidden' items
            if isinstance(key,str) and key.startswith('_'):
                continue
            
            # recurse into other dict types
            if isinstance(value,OrderedDict):
                if not value:
                    val = '\n'
                else:
                    try:
                        val = value.__str2(indent+new_indent)
                    except RuntimeError: # recursion limit
                        val = ''
                    except:
                        val = value.__str__(indent+new_indent)                                    
                        
            # everything else
            else:
                val = str(value) + '\n'
                
            # this key-value, indented
            args+= indent + str(key) + ' : ' + val
            
        return args     

    def clear(self):
        """ Empties a dictionary
            
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """        
        
        try:
            for node in self._map.values():
                del node[:]
            root = self._root
            root[:] = [root, root, None]
            self._map.clear()
        except AttributeError:
            pass
        self.__dict__.clear()
        
    def get(self,k,d=None):
        """ Returns the values from k
            
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            k
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """         
        return self.__dict__.get(k,d)
        
    def has_key(self,k):
        """ Checks if the dictionary has the key, k
            
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            k
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """             
        return k in self.__dict__

    # allow override of iterators
    __iter = __iter__
    __getitem2 = OrderedDict.__getattribute__ 

    def keys(self):
        """ Returns a list of keys
            
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """         
        return list(self.__iter_basic__())
    
    def values(self):
        """ Returns all values inside the Data() class.
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            values
    
            Properties Used:
            N/A    
        """             
        return [self[key] for key in self.__iter_basic__()]
    
    def items(self):
        """ Returns all the items inside the data class
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            values
    
            Properties Used:
            N/A    
        """          
        return [(key, self[key]) for key in self.__iter_basic__()]
    
    def iterkeys(self):
        """ Returns all the keys which may be iterated over
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """         
        return self.__iter_basic__()
    
    def itervalues(self):
        """ Finds all the values that can be iterated over.
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """          
        for k in self.__iter_basic__():
            yield self[k]
    
    def iteritems(self):
        """ All items that may be iterated over
            
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """          
        for k in self.__iter():
            yield (k, self[k])   


# for rebuilding dictionaries with attributes
def _reconstructor(klass,items):
    """ For rebuilding dictionaries with attributes
        
        Assumptions:
        N/A

        Source:
        N/A

        Inputs:
        N/A

        Outputs:
        N/A

        Properties Used:
        N/A    
    """        
    self = DataOrdered.__new__(klass)
    DataOrdered.__init__(self,items)
    return self