
import multiprocessing as mp
import numpy as np
import time
import sys


class Evaluator(object):
    def __init__(self,function,args,outbox):
        self.function = function
        self.outbox = outbox
        self.args   = args
        
    def __call__(self,inputs):
        index,inputs = inputs
        output = self.function(inputs,self.args)
        self.outbox.put([index,output])
        return
        