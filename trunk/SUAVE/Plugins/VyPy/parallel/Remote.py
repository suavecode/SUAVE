
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from Task           import Task
from ShareableQueue import ShareableQueue
from GlobalManager  import GlobalManager

import multiprocessing as mp

# ----------------------------------------------------------------------
#   Remote
# ----------------------------------------------------------------------

class Remote(object):
    
    def __init__(self,inbox,outbox=None):
        self.inbox  = inbox
        self.outbox = outbox
        manager = mp.Manager()
        self.local_queue = ShareableQueue(manager=manager)
    
    def put(self,task):
        self.inbox.put( task )
        
    def get(self,block=True,timeout=None):
        
        if self.outbox is None:
            raise AttributeError, 'no outbox to query'

        task = self.outbox.get(block,timeout)
        
        return task
    
    
    def __call__(self,inputs,function=None):
        
        # could get expensive with lots of __call__
        local_queue = self.local_queue
        
        if not isinstance(inputs,Task):
            this_task = Task( inputs   = inputs ,
                              function = function ,
                              outbox   = local_queue )
        else:
            this_task = inputs
            this_task.outbox = local_queue
        
        self.put(this_task)

        task = local_queue.get(block=True)
        
        outputs = task.outputs
        
        return outputs