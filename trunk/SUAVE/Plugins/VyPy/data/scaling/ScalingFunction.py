
class ScalingFunction(object):
    def __init__(self):
        pass
    
    def set_scaling(self,other):
        return other
    def unset_scaling(self,other):
        return other
    
    def set_scaling_gradient(self,other):
        raise NotImplementedError
    def unset_scaling_gradient(self,other):
        raise NotImplementedError
    
    # operator overloading
    def __rdiv__(self,other):
        return self.set_scaling(other)    
    def __rmul__(self,other):
        ## TODO
        #if isinstance(other,Gradient):
            #return self.unset_scaling_gradient(other)
        return self.unset_scaling(other)
    __mul__ = __rmul__
    __div__ = __rdiv__
    __truediv__ = __div__
    
    __array_priority__ = 100