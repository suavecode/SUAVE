
from Ordered_Bunch import Ordered_Bunch
import copy

class Indexable_Bunch(Ordered_Bunch):
    """ An Ordered_Bunch with list-style access 
    """
    
    def __getitem__(self,k):
        if isinstance(k,int):
            try:           
                return self[ self.keys()[k] ]
            except IndexError:
                raise IndexError(k)
        else:
            return super(Indexable_Bunch,self).__getitem__(k)
    
    def __setitem__(self,k,v):
        if isinstance(k,int):
            try:
                self[ self.keys()[k] ] = v
            except IndexError:
                self[str(k)] = v
        else:
            super(Indexable_Bunch,self).__setitem__(k,v)
            
    def __delitem__(self,k):
        if isinstance(k,int):
            try:
                del self[ self.keys()[k] ]
            except IndexError:
                raise IndexError(k)
        else:
            super(Indexable_Bunch,self).__delitem__(k)
    
    def index(self,key):
        if isinstance(key,int):
            index = range(len(self))[key]
        else:
            index = self.keys().index(key)
        return index
    
    def append(self,value,key=None):
        if key is None: key = value.tag
        if key in self: raise KeyError, 'key "%s" already exists' % key
        self[key] = value
            
    def insert(self,index,key,value):
        """ Indexable_Bunch.insert(index,key,value)
            insert key and value before index
            index can be an integer or key name
        """
        # potentially expensive....
        
        # original length
        len_self = len(self)
        
        # add to dictionary
        self[key] = value
        
        # find insert index number
        index = self.index(index)
        
        # done if insert index is greater than list length
        if index >= len_self: return
        
        # insert into index list
        indeces = range(0,len_self)
        indeces.insert(index,len_self)
        
        # repack dictionary
        keys = self.keys()
        values = self.values()
        self.clear()
        for i in indeces: self[keys[i]] = values[i]
    
    def swap(self,key1,key2):
        """ Indexable_Bunch.swap(key1,key2)
            swap key locations 
        """
        # potentially expensive....
        
        # get swapping indeces
        index1 = self.index(key1)
        index2 = self.index(key2)
        
        # list of all indeces
        indeces = range(0,len(self))
        
        # replace index1 with index2
        indeces.insert(index1,index2)
        del(indeces[index1+1])
        
        # replace index2 with index1
        indeces.insert(index2,index1)
        del(indeces[index2+1])
        
        # repack dictionary
        keys = self.keys()
        values = self.values()
        self.clear()
        for i in indeces: self[keys[i]] = values[i]        
        