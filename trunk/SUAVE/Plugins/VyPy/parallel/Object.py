
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------


import sys, copy, inspect, types

from Operation import Operation
from Service   import Service
from Remote    import Remote
from Task      import Task
from ShareableQueue import ShareableQueue
from KillTask  import KillTask


PROXY_IGNORE=[
    '__init__',
    '__reduce__',
    '__getstate__',
    '__setstate__',
    '_OrderedDict__update',
    '__getattr__',
    '__setattr__',
    '__delattr__',
    '__getattribute__',
]

PROXY_BUILTIN = [
    #'__new__',
    #'__init__',
    '__cmp__',
    '__pos__',
    '__neg__',
    '__invert__',
    '__index__',
    '__nonzero__',
    #'__getattr__',
    #'__setattr__',
    #'__delattr__',
    #'__getattribute__',
    '__getitem__',
    '__setitem__',
    '__delitem__',
    '__iter__',
    '__contains__',
    '__call__',
    '__enter__',
    '__exit__',
    '__del__',
    #    '__getstate__',
    #    '__setstate__',
]


# ----------------------------------------------------------------------
#   Object Service Proxy
# ----------------------------------------------------------------------

class Object(object):

    def __init__(self,obj):
        
        obj = copy.deepcopy(obj)

        self.name = str( type(obj) ) # todo - make prettier
        
        _object = ObjectService(obj)
        self._remote = _object.remote()
        
        self.dec_methods(obj)
        
    def dec_methods(self,obj):

        for name, meth in inspect.getmembers(obj):
            type_str = str(type(meth))
            
            if 'method'   not in type_str and \
               'function' not in type_str: continue
            elif name in PROXY_IGNORE: continue

            if name in PROXY_BUILTIN:
                pxy_name = '__proxy' + name
            else:
                pxy_name = name
            
            if hasattr(self,pxy_name): continue
    
            monkey_patch = ProxyClassMethod(self,name)

            setattr(self, pxy_name, monkey_patch )
            
        #: for each method
        
    def put(self,obj):
        self.__send__('put',obj)
        
    def get(self):
        obj = self.__send__('get')
        return obj
        
        
    def __send__(self,method_name,*args,**kwarg):

        inputs = [method_name,args,kwarg]
        
        outputs = self._remote(inputs)

        return outputs


    def __cmp__(self, other):
        """ self == other, self > other, etc. 
            Called for any comparison
        """
        return self.__proxy__cmp__(other)

    def __pos__(self):
        """ +self 
            Unary plus sign
        """
        return self.__proxy__pos__()

    def __neg__(self):
        """ -self Unary minus sign
        """
        return self.__proxy__neg__()

    def __invert__(self):
        """ ~self 
            Bitwise inversion
        """
        return self.__proxy__invert__()

    def __index__(self):
        """ x[self]
            Conversion when object is used as index
        """
        return self.__proxy__index__()

    def __nonzero__(self):
        """ bool(self) 
            Boolean value of the object
        """
        return self.__proxy__nonzero__()

    #def __getattr__(name):
        #""" self.name 
            #Accessing nonexistent attribute
        #"""
        #return self.__proxy__getattr__(name)

    #def __setattr__(self, name, val):
        #""" self.name = val 
            #Assigning to an attribute
        #"""
        #return self.__proxy__setattr__(name,val)

    #def __delattr__(self, name):
        #""" del self.name
            #Deleting an attribute
        #"""
        #return self.__proxy__delattr__(name)

    #def __getattribute__(self, name):
        #""" self.name 
            #Accessing any attribute
        #"""
        #return self.__proxy__getattribute__(name)

    def __getitem__(self,key):
        """ self[key] 
            Accessing an item using an index
        """
        return self.__proxy__getitem__(key)

    def __setitem__(self, key, val):
        """ self[key] = val
            Assigning to an item using an index
        """
        return self.__proxy__setitem__(key,val)

    def __delitem__(self, key):
        """ del self[key]
            Deleting an item using an index
        """
        return self.__proxy__delitem__(key)

    def __iter__(self):
        """ for x in self 
            Iteration
        """
        return self.__proxy__iter__()

    def __contains__(self, value):
        """ value in self, value not in self
            Membership tests using in
        """
        return self.__proxy__contains__(value)

    def __call__(self, *args,**kwarg):
        """ self(*args,**kwarg) 
            Calling an instance
        """
        return self.__proxy__call__(value)

    def __enter__(self, *args,**kwarg):
        """ with self as x: 
            with statement context managers
        """
        return self.__proxy__enter__(*args,**kwarg)

    def __exit__(self, exc, val, trace):
        """ with self as x:
            with statement context managers
        """
        return self.__proxy__exit__(exc,val,trace)
    
    def __del__(self):
        self.__proxy__del__()
        self._remote.put(KillTask)
        self._remote.inbox.join()





# ----------------------------------------------------------------------
#   Object Service
# ----------------------------------------------------------------------

class ObjectService(Service):

    def __init__(self,obj,name='Object'):
        
        Service.__init__(self)
        
        self.name    = name
        self._object = obj
        
        self.start()
        
    def function(self,inputs):

        obj = self._object
        meth,args,kwarg = inputs
        
        if meth == 'get':
            outputs = obj
        elif meth == 'put':
            obj = args[0]
            outputs = None
        else:
            meth = getattr(obj,meth)
            outputs = meth(*args,**kwarg)

        return outputs
    




# ----------------------------------------------------------------------
#   Helper Class
# ----------------------------------------------------------------------

class ProxyClassMethod(object):
    def __init__(self,Obj,method_name):
        self._Object     = Obj
        self.method_name = method_name
    def __call__(self,*args,**kwarg):
        return self._Object.__send__(self.method_name,*args,**kwarg)
    def __get__(self): # schneaky schneaky
        return self


# ----------------------------------------------------------------------
#   Tests
# ----------------------------------------------------------------------

class TestClass(object):
    def __init__(self,a=1,b=2):
        self.a = a
        self.b = b
    
    def testfunc(self,inputs):
        print inputs , '!!!'
        return True
    

if __name__ == '__main__':

    a = dict()
    a['a1'] = 23
    a['b4'] = 56

    print a['a1']

    b = Object(a)
    
    print b['a1']
    
    print b['b4']
    
    b['a1'] = 51
    b['44'] = 'hello'
    
    print b['a1']
    print b['44']
    
    
    c = TestClass()
    c.a = 23
    
    print c.a
    print c.testfunc('Hello')
    
    
    d = Object(c)
    
    d.a = 24
    
    print d.a
    print d.testfunc('World')
    
    print 'success!'