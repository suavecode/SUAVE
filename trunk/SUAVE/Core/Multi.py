

class Evaluator(object):
    def __init__(self,function,args):
        self.function = function
        self.args   = args
        
    def __call__(self,inputs):
        output = self.function(inputs,self.args)
        return output        