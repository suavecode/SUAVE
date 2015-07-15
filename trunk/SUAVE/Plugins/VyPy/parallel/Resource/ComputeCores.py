
from Resource import Resource
from VyPy.parallel import Service
from VyPy.exceptions import ResourceException, ResourceWarning
import sys

class ComputeCores(Resource):
    
    def __init__(self,max_cores=1,name='Cores'):
        
        Resource.__init__(self)
        
        cores = self.manager.BoundedSemaphore(max_cores)
        
        in_service = InService(     \
            inbox = self.checkinbox ,
            cores = cores           ,
            name  = name+'_checkin' ,
        )        
        
        out_service = OutService(    \
            inbox = self.checkoutbox ,
            cores = cores            ,
            name  = name+'_checkout' ,
        )        
        
        self.name = name
        self.max_cores   = max_cores
        self.cores       = cores
        self.in_service  = in_service
        self.out_service = out_service
        
        self.start()
        
        
class OutService(Service):
    def __init__(self,inbox,cores,name):
        Service.__init__(self,inbox=inbox,name=name)
        self.cores = cores
        #self.verbose = False
    
    def function(self,request):
        
        n_cores   = request
        semaphore = self.cores
                
        #raise ResourceException , 'attemted to checkout more compute cores than available: %i>%i' % (ncores,semaphore.maxvalue)
        
        # checkout
        for c in range(n_cores):
            semaphore.acquire(blocking=True)
            
        return
    
class InService(Service):
    def __init__(self,inbox,cores,name):
        Service.__init__(self,inbox=inbox,name=name)
        self.cores = cores
        #self.verbose = False
    
    def function(self,request):
            
        n_cores   = request
        semaphore = self.cores
        
        # checkin
        for c in range(n_cores):
            semaphore.release()
            #print 'Warning: attempted to check-in more processes than available'
        
        return
    
    
        
        
if __name__ == '__main__':
    
    resource = ComputeCores(max_cores=4)
    print resource
    
    gate = resource.gate(default=2)
    print gate
    
    gate.check_out(2)
    gate.check_out(2)
    gate.check_in(2)
    
    gate.check_out(2)
    gate.check_in(2)
    gate.check_in(2) # baad
    
    with gate:
        print 'acquired'
    print 'released'
    
    gate.check_out(2)
    gate.check_out(2)
    # gate.check_out(2) # block    
    
    print 'done!'