'''
Define the optimization class.

Rick Fenrich 8/9/16
'''

import numpy as np

#class Algorithm():
class Optimization_Algorithm():
    def __init__(self):
        self.name = 'none'
        
        self.max_iterations = 50
        self.max_function_evaluations = 1000
        
        self.convergence_tolerance = 1e-6
        self.constraint_tolerance = 1e-6
        
        self.difference_interval = 1e-6

