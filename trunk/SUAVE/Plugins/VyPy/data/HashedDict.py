
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from IndexableDict import IndexableDict
    
from make_hashable import make_hashable


# ----------------------------------------------------------------------
#   Hashable Dict
# ----------------------------------------------------------------------

class HashedDict(IndexableDict):
    """ An indexable dictionary that permits typically unhashable keys, 
        such as lists and other dictionaries
    """
    
    def __getitem__(self,k):
        _k = make_hashable(k) 
        return super(HashedDict,self).__getitem__(_k)
        #raise KeyError , ('Key does not exist: %s' % k)
        
    def __setitem__(self,k,v):
        _k = make_hashable(k)
        super(HashedDict,self).__setitem__(_k,v)
        
    def __delitem__(self,k):
        _k = make_hashable(k)
        super(HashedDict,self).__delitem__(_k)
        
    def __contains__(self,k):
        _k = make_hashable(k)
        return super(HashedDict,self).__contains__(_k)
    
    def has_key(self,k):
        _k = make_hashable(k)
        return super(HashedDict,self).has_key(_k)


# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------

if __name__ == '__main__':
    
    # tests
    cache = HashedDict()
    cache['a'] = 1
    cache[[1,2,3]] = 2
    print 'should be True:' , cache.has_key([1,2,3])
    print 'should be True:' , [1,2,3] in cache
    del cache[[1,2,3]]
    print 'should be False:' , cache.has_key([1,2,3])
    cache[[1,2,5]] = 5
            
            
    import pickle
        
    d = pickle.dumps(cache)
    p = pickle.loads(d)
    
    print ''
    print p[1]
    print 'should be True:' , [1,2,5] in cache
    
    