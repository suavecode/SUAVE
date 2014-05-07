
# ----------------------------------------------------------------------
#   Object, with some mods
# ----------------------------------------------------------------------

class Object(object):
    
    # implement descriptor protocol for instances
    # mega expensive...
    def __getattribute__(self,k):
        try:
            return object.__getattribute__(self,k).__get__(self,type(self))
        except AttributeError:
            return object.__getattribute__(self,k)
    
    def __setattr__(self,k,v):
        try:
            object.__getattribute__(self,k).__set__(self,v)
        except AttributeError:
            object.__setattr__(self,k,v)
    
    def __delattr__(self,k):
        try:
            object.__getattribute__(self,k).__delete__(self)
        except AttributeError:
            object.__delattr__(self,k)
          
            
# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------            
            
if __name__ == '__main__':
    
    class TestDescriptor(object):
        def __init__(self,x):
            self.x = x
        
        def __get__(self,obj,kls=None):
            print '__get__:' , self.x
            return self.x
        
        def __set__(self,obj,val):
            print '__set__:' , val
            self.x = val
        
    class TestObject(Object):
        def __init__(self,x):
            self.x = TestDescriptor(x)
        
    
    o = TestObject([1,2,3])
    
    print o.x
    
    o.x = [3,4,5]
            