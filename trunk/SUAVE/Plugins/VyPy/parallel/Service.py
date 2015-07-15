
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import os, sys, shutil, traceback

import multiprocessing as mp
import multiprocessing.queues

from Task           import Task
from Operation      import Operation
from KillTask       import KillTask
from ShareableQueue import ShareableQueue
from Remote         import Remote

from VyPy.tools import redirect, check_pid
from VyPy.data import HashedDict
 
# ----------------------------------------------------------------------
#   Process
# ----------------------------------------------------------------------

class Service(mp.Process):
    """ preloaded func before task specified func
    """
    def __init__( self, function=None, inbox=None, outbox=None, 
                  name=None, verbose=False ):
        """ VyPy.parallel.Service()
            starts a service on a new thread
            this service will look for a Task() in the inbox queue, 
            execute the task, and place the task in the outbox queue
            
            Inputs:
                function  - (optional) the function to call with task inputs
                inbox     - (optional) the inbox queue, of VyPy.parallel.Queue()
                outbox    - (optional) the outbox queue, of VyPy.parallel.Queue()
                name      - (optional) the name of the service, for verbose logging
                verbose   - (optional) True - print service events to stdout
                                       False(default) - quiet service
                                       <filename> - print service events to filename
            
            Methods:
                start() - start the service.  this is *not* called automatically
                remote() - get a remote for this service
                function(inputs) - the function called for each task
                    note: will not be called if task includes a function
                
            The Service class can and should be extended for important services.
            
            Some ways this is useful:
                Define the service function(), inboxes, and outboxes
                Define service constants (as attributes)
                Call start() at the end of object instantiation
            
            
        """

        # super init
        mp.Process.__init__(self)      
        self.name = name or self.name
        
        # initialize function carrier
        if function and not isinstance(function,Operation):
            function = Operation(function)
        
        # store class vars
        self.inbox     = inbox or ShareableQueue()
        self.outbox    = outbox
        self.function  = function or self.function
        self.daemon    = True
        self.parentpid = os.getpid()
        self.verbose   = verbose
        
        # Remote.__init__(self,self.inbox,self.outbox)       ???
        
        #self.start()
        
    def function(self,inputs):
        raise NotImplementedError
        
    def __func__(self,inputs,function):
        if self.verbose: print '%s: Starting task' % self.name; sys.stdout.flush()
        outputs = function(inputs)
        return outputs
    
    def start(self):
        mp.Process.start(self)

    def run(self):
        """ Service.run()
            the service executes the following steps until killed - 
                1. check that parent process is still alive
                2. check for a new task from inbox
                3. execute the task
                4. if the task is succesful, store the result 
                   if the task is unsuccesful, store the exception traceback
                5. put the task into an outbox
                6. continue until receive KillTask() or parent process dies
        """
        try:
            if isinstance(self.verbose,str):
                with redirect.output(self.verbose,self.verbose):
                    self.__run__()
            else:
                self.__run__()
        except:
            sys.stderr.write( '%s: Unhandled Exception \n' % self.name )
            sys.stderr.write( traceback.format_exc() )
            sys.stderr.write( '\n' )
            sys.stderr.flush()

    def __run__(self):
        """ Service.__run__()
            runs the service cycle
        """
            
        # setup
        name = self.name
        if self.verbose: print 'Starting %s' % name; sys.stdout.flush()    
        
        # --------------------------------------------------------------
        #   Main Cycle - Continue until Death
        # --------------------------------------------------------------
        while True:
            
            # --------------------------------------------------------------
            #   Check parent process status
            # --------------------------------------------------------------
            if not check_pid(self.parentpid):
                break
            
            # --------------------------------------------------------------
            #   Check inbox queue
            # --------------------------------------------------------------
            # try to get a task
            try:
                this_task = self.inbox.get(block=True,timeout=1.0)
            # handle exceptions
            except ( mp.queues.Empty,     # empty queue
                     EOFError, IOError ): # happens on python exit
                continue
            # raise exceptions for python exit
            except (KeyboardInterrupt, SystemExit):
                raise
            # print any other exception
            except Exception as exc:
                if self.verbose: 
                    sys.stderr.write( '%s: Get Failed \n' % name )
                    sys.stderr.write( traceback.format_exc() )
                    sys.stderr.write( '\n' )
                    sys.stderr.flush()
                continue
            
            # report
            if self.verbose: print '%s: Got task' % name; sys.stdout.flush()
            
            # --------------------------------------------------------------
            #   Execute Task
            # --------------------------------------------------------------
            try:
                # check task object type
                if not isinstance(this_task,Task):
                    #raise Exception, 'Task must be of type VyPy.Task'
                    this_task = Task(this_task,self.function)
                
                # unpack task
                this_input    = this_task.inputs
                this_function = this_task.function or self.function
                this_owner    = this_task.owner
                this_folder   = this_task.folder
                
                # --------------------------------------------------------------
                #   Check for kill signal
                # --------------------------------------------------------------
                if isinstance(this_input,KillTask.__class__):
                    break
                
                # report
                if self.verbose: print '%s: Inputs = %s, Operation = %s' % (name,this_input,self.function); sys.stdout.flush()
                
                # --------------------------------------------------------------
                #   Call Task Function
                # --------------------------------------------------------------
                # in the folder it was created
                with redirect.folder(this_folder):
                    # report
                    if self.verbose: print os.getcwd(); sys.stdout.flush()
                    
                    # the function call!
                    this_output = self.__func__(this_input,this_function)
                
                # report
                if self.verbose: print '%s: Task complete' % name; sys.stdout.flush()                
                
                # make sure we get std's
                sys.stdout.flush()
                sys.stderr.flush()
            
            # --------------------------------------------------------------
            #   Task Exception Catching
            # --------------------------------------------------------------
            # system exits
            except (KeyboardInterrupt, SystemExit):
                raise
            # all other exceptions
            except Exception as exc:
                trace_str = traceback.format_exc()
                if self.verbose: 
                    sys.stderr.write( '%s: Task Failed \n' % name )
                    sys.stderr.write( trace_str )
                    sys.stderr.write( '\n' )
                    sys.stderr.flush()
                exc.args = (trace_str,)
                this_output = exc
                
            #: end try task
            
            # --------------------------------------------------------------
            #   Wrap Up Task
            # --------------------------------------------------------------
            
            # store output
            this_task.outputs = this_output            
            
            # pick outbox
            this_outbox = this_task.outbox or self.outbox
            # avoid serialization error with managed outboxes
            this_task.outbox = None 
            
            # put task
            if this_outbox: this_outbox.put(this_task)
            
            # end joinable inbox task
            self.inbox.task_done()
            
        #: end while alive
        
        # --------------------------------------------------------------
        #   End of Process
        # --------------------------------------------------------------
        
        # end joinable inbox task
        self.inbox.task_done()
        
        # report
        if self.verbose: print '%s: Ending' % name; sys.stdout.flush()
        
        return
    
    def put(self,task):
        self.inbox.put( task )            
        
    def get(self,block=True,timeout=None):
        
        if self.outbox is None:
            raise AttributeError, 'no outbox to query'

        task = self.outbox.get(block,timeout)
        
        return task
        
    def remote(self):
        return Remote(self.inbox,self.outbox)
    
    
# ----------------------------------------------------------------------
#   Tests
# ----------------------------------------------------------------------    

def test_func(x):
    y = x*2.
    print x, y
    return y  

def fail_func(x):
    raise Exception , ('bork!' , x)


if __name__ == '__main__':
    
    inbox = ShareableQueue()
    outbox = ShareableQueue()
    
    function = test_func
    
    service = Service(function, inbox, outbox,
                      name='TestService',verbose=False)
    
    service.start()
    
    inbox.put(10.)
    inbox.put(20.)
    
    print outbox.get().outputs
    print outbox.get().outputs
    
    remote = service.remote()
    
    print remote(30.)
    
    print 'this will print an exception traceback:'
    task = Task(inputs=20,function=fail_func)
    print remote(task)
    
    inbox.put(KillTask)
    
    inbox.join()
    
    print 'test done!'