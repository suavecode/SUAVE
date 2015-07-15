
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import os
from Operation import Operation

# ----------------------------------------------------------------------
#   Task
# ----------------------------------------------------------------------
    
class Task(object):
    def __init__(self,inputs,function=None,outbox=None):
        
        if function and not isinstance(function,Operation):
            function = Operation(function)
            
        self.inputs   = inputs
        self.function = function
        self.outputs  = None 
        self.owner    = os.getpid()
        self.folder   = os.getcwd()
        self.outbox   = outbox