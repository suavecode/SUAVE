
# ----------------------------------------------------------------------
#   Operation
# ----------------------------------------------------------------------

class Operation(object):
    
    def __init__(self, function=None, gate=None):
        self.function = function or self.function
        self.gate = gate or DummyGate()
        if not hasattr(self.gate,'__enter__') and not hasattr(self.gate,'__exit__'):
            raise AttributeError, 'gate must support context management'
        
    def function(self):
        raise NotImplementedError
        
    def __call__(self, *arg, **kwarg):  
        """ makes object callable
            will wait for gate to open
        """
        with self.gate:
            outputs = self.function(*arg, **kwarg)
            
        return outputs
    
    def __repr__(self):
        if hasattr(self.function,'__name__'):
            name = self.function.__name__
        else:
            name = repr(self.function)
        return '<Operation %s>' % name

    # pickling
    #def __getstate__(self):
        #dict = self.__dict__.copy()
        #data_dict = cloudpickle.dumps(dict)
        #return data_dict

    #def __setstate__(self,data_dict):
        #self.__dict__ = pickle.loads(data_dict)
        #return    
        
    
    
class DummyGate(object):
    def __enter__(self):
        return
    def __exit__(self, exc_type, exc_value, traceback):
        return


    
# ----------------------------------------------------------------------
#   Tests
# ----------------------------------------------------------------------
    
def test_func(x):
    y = x*2.
    print x, y
    return y

if __name__ == '__main__':
    
    function = test_func
    function = Operation(function)
    print function(10.)