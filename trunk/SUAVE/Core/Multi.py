
import multiprocessing as mp
import numpy as np
import time
import sys


class Evaluator(object):
    def __init__(self,function,outbox):
        self.function = function
        self.outbox = outbox
        
    def __call__(self,inputs):
        index,inputs = inputs
        output = self.function(inputs)
        self.outbox.put([index,output])
        return
        