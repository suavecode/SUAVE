
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from OrderedDict import OrderedDict


# ----------------------------------------------------------------------
#   Indexable Dictionary
# ----------------------------------------------------------------------

class IndexableDict(OrderedDict):
    """ An OrderedDict with list-style access 
    """
        
    def __getitem__(self,k):
        if not isinstance(k,int):
            return super(IndexableDict,self).__getitem__(k)
        else:
            return super(IndexableDict,self).__getitem__(self.keys()[k])
    
    def __setitem__(self,k,v):
        if not isinstance(k,int):
            super(IndexableDict,self).__setitem__(k,v)
        else:
            super(IndexableDict,self).__setitem__( self.keys()[k], v )
    
    def __delitem__(self,k):
        if not isinstance(k,int):
            return super(IndexableDict,self).__delitem__(k)
        else:
            return super(IndexableDict,self).__delitem__(self.keys()[k])
    
    # iterate on values, not keys
    def __iter__(self):
        return super(IndexableDict,self).itervalues()

    def index(self,key):
        if isinstance(key,int):
            index = range(len(self))[key]
        else:
            index = self.keys().index(key)
        return index  
    
    def insert(self,index,key,value):
        """ IndexableDict.insert(index,key,value)
            insert key and value before index
            index can be an integer or key name
        """
        # potentially expensive....
        # clears dictionary in process...
        
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
        indeces.insert(index,len_self) # 0-based indexing ...
        
        # repack dictionary
        keys = self.keys()
        values = self.values()
        self.clear()
        for i in indeces: self[keys[i]] = values[i]
    
    def swap(self,key1,key2):
        """ IndexableDict.swap(key1,key2)
            swap key locations 
        """
        # potentially expensive....
        # clears dictionary in process...
        
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
        
        
# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------        
        
if __name__ == '__main__':
    
    o = IndexableDict()
    o['x'] = 'hello'
    o['y'] = 1
    o['z'] = [3,4,5]
    o['t'] = IndexableDict()
    o['t']['h'] = 20
    o['t']['i'] = (1,2,3)

    print o

    import pickle

    d = pickle.dumps(o)
    p = pickle.loads(d)
    
    print ''
    print p    
    
    o[3][0] = 'changed'
    p.update(o)

    print ''
    print p
    