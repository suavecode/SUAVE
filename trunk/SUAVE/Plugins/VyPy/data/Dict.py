
# ----------------------------------------------------------------------
#   Dictionary (with some upgrades)
# ----------------------------------------------------------------------

class Dict(dict):

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
    
    # new keys by wild card integer
    def next_key(self,key_wild):
        """ Dict.next_key(key_wild):
            finds the next index to use on a indexed key and applies it to key_wild
            key_wild is a string containing '%i' to indicate where to increment
            the key
        """
        
        if '%i' not in key_wild:
            return key_wild
        
        ksplit = key_wild.split('%i')
        
        keys = []
        for k in keys():
            try:
                i = int( k.lstrip(ksplit[0]).rstrip(ksplit[1]) )
                keys.append(i)
            except:
                pass
            
        if keys:
            key_index = max(keys)+1
        else:
            key_index = 0
        
        key = key_wild % (key_index)
        
        return key
    
    # allow override of iterators
    __iter = dict.__iter__
    
    def keys(self):
        """Dict.keys() -> list of keys in the dictionary"""
        return list(self.__iter())
    
    def values(self):
        """Dict.values() -> list of values in the dictionary"""
        return [self[key] for key in self.__iter()]
    
    def items(self):
        """Dict.items() -> list of (key, value) pairs in the dictionary"""
        return [(key, self[key]) for key in self.__iter()]
    
    def iterkeys(self):
        """Dict.iterkeys() -> an iterator over the keys in the dictionary"""
        return self.__iter()
    
    def itervalues(self):
        """Dict.itervalues -> an iterator over the values in the dictionary"""
        for k in self.__iter():
            yield self[k]
    
    def iteritems(self):
        """od.iteritems -> an iterator over the (key, value) items in the dictionary"""
        for k in self.__iter():
            yield (k, self[k])    

    # prettier printing
    def __repr__(self):
        """ Invertible* string-form of a Dict.
        """
        keys = self.keys()
        args = ', '.join(['%s=%r' % (key, self[key]) for key in keys if not key.startswith('_')])
        return '%s(%s)' % (self.__class__.__name__, args)

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
            if isinstance(value,Dict):
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


# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------

if __name__ == '__main__':

    o = Dict()
    o['x'] = 'hello'
    o['y'] = 1
    o['z'] = [3,4,5]
    o['t'] = Dict()
    o['t']['h'] = 20
    o['t']['i'] = (1,2,3)
    
    print o
    
    import pickle
    
    d = pickle.dumps(o)
    p = pickle.loads(d)
    
    print ''
    print p
    
    o['t']['h'] = 'changed'
    p.update(o)
    p['t'].update(o)

    print ''
    print p
    
    
    
    
# ----------------------------------------------------------------------
#   Gravetart
# ----------------------------------------------------------------------
    
    
        ## implement descriptor protocol for items
        #def __getitem__(self,k):
            #try:
                #return super(Dict,self).__getitem__(k).__get__(self,type(self))
            #except AttributeError:
                #return super(Dict,self).__getitem__(k)
            #except KeyError:
                #raise KeyError(k)
    
        #def __setitem__(self,k,v):
            #try:
                #super(Dict,self).__getitem__(k).__set__(self,v)
            #except AttributeError:
                #super(Dict,self).__setitem__(k,v)
            #except KeyError:
                #raise KeyError(k)
    
        #def __delitem__(self,k):
            #try:
                #super(Dict,self).__getitem__(k).__del__(self)
            #except AttributeError:
                #super(Dict,self).__delitem__(k)
            #except KeyError:
                #raise KeyError(k)    
    


    #class TestDescriptor(object):
        #def __init__(self,x):
            #self.x = x

        #def __get__(self,obj,kls=None):
            #print '__get__'
            #print type(obj), type(self)
            #print self.x
            #return self.x

        #def __set__(self,obj,val):
            #print '__set__'
            #print type(obj), type(self)
            #print val
            #self.x = val

    #class TestObject(Dict):
        #def __init__(self,c):
            #self.c = c

    #o = TestObject(555)
    #o['x'] = TestDescriptor([1,2,3])
    #o['y'] = 1
    #o.desc = TestDescriptor([5,7,8])

    #print ''
    #print o['x']
    #print o['y']
    ##print o.desc
    ##print o.c
    #print o

    #print ''
    #o['x'] = [3,4,5]


    #import pickle

    #d = pickle.dumps(o)
    #p = pickle.loads(d)

    #print ''
    #print p['x']
    #print p['y']
    #print o.c
    #print o.desc