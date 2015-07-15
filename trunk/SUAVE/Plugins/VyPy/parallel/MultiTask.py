
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import os, sys, shutil

import multiprocessing as mp
from Service   import Service
from Remember  import Remember
from Object    import Object
from Operation import Operation
from Task      import Task
from KillTask  import KillTask
from ShareableQueue import ShareableQueue

from VyPy.exceptions import FailedTask
from VyPy.data import HashedDict
from VyPy.tools import redirect

from VyPy.tools.arrays import numpy_isloaded, array_type, matrix_type


## TODO
# add base folder

# ----------------------------------------------------------------------
#   MultiTask
# ----------------------------------------------------------------------

class MultiTask(object):
    
    def __init__( self, function=None, copies=0, 
                  inbox=None, outbox=None,
                  name='MultiTask', verbose=False ):
        # todo: priority, save cache, setup
        
        # initialize function carrier
        if function:
            if isinstance(function,Remember):
                if not isinstance(function.__cache__,Object):
                    function.__cache__ = Object(function.__cache__)
            elif isinstance(function,Service):
                raise Exception, 'cannot create MultiTask() with a Service()'
            elif not isinstance(function,Operation):
                function = Operation(function)
        
        # initialize queues
        inbox  = inbox  or ShareableQueue()
        outbox = outbox or ShareableQueue()
        
        # check numer of copies
        if copies < 0: copies = max(mp.cpu_count()+(copies+1),0)
        
        # store
        self.function   = function
        self.name       = name
        self.inbox      = inbox
        self.outbox     = outbox
        self.copies     = copies 
        self.verbose    = verbose
        
        # auto start
        if function:
            self.start()
        
    
    def start(self):
        if isinstance(self.verbose,str):
            with redirect.output(self.verbose,self.verbose):
                self.__start__()
        else:
            self.__start__()
        
    
    def __start__(self):
        
        function = self.function
        name     = self.name
        inbox    = self.inbox
        outbox   = self.outbox
        copies   = self.copies
        verbose  = self.verbose
        
        # report
        if verbose: print '%s: Starting %i Copies' % (name,copies); sys.stdout.flush()
        
        # initialize services
        nodes = []
        for i in range(copies):
            this_name = '%s - Node %i'%(name,i)
            service = Service( function, inbox, outbox, this_name, verbose ) 
            service.start()
            nodes.append(service)
        #: for each node        
                    
        # store
        self.nodes = nodes
        
    def __func__(self,inputs):
        outputs = self.function(inputs)
        return outputs
        
    def __call__(self,x):
        
        # check for multiple inputs
        if isinstance(x,(list,tuple)): 
            xx = x   # multiple inputs
            ff = [None] * len(xx)
            f = [None]
        elif numpy_isloaded and isinstance(x,(array_type,matrix_type)):
            xx = x   # multiple inputs
            ff = np.zeros_like(x)
            f = np.array([None])
        else:
            xx = [x] # single input
            ff = [None]
            f = None
            
        # submit input
        for i,x in enumerate(xx):
            this_task = Task( inputs = x )
            self.put(this_task)
                
        # wait for output, if there's an outbox
        if self.outbox is None: return
        to = self.get()
        
        # sort output
        for i,x in enumerate(xx):
            ff[i] = to[x].outputs
            
        # format for output
        if f is None: f = ff[0] # single input
        else:         f = ff    # multiple input
                
        # done
        return f
                
    def put(self,task):
        self.inbox.put( task )            
        return
        
    def get(self):
        
        if self.outbox is None:
            raise AttributeError, 'no outbox to query'
        
        # for sorting results
        results = HashedDict()
        
        # wait for all jobs to process
        self.inbox.join()
        
        # pull results
        while not self.outbox.empty():
            task = self.outbox.get()
            x = task.inputs
            results[x] = task
        
        return results
    
    def remote(self):
        return Remote(self.inbox,self.outbox)
    
    def __del__(self):
        try:
            for p in self.nodes:
                self.inbox.put(KillTask)
            self.inbox.join()
        except:
            pass

