
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from VyPy.exceptions import ResourceException, ResourceWarning
from VyPy.parallel import KillTask, Remote, ShareableQueue, Service

import multiprocessing as mp


# ----------------------------------------------------------------------
#   Resource
# ----------------------------------------------------------------------

class Resource(object):
    
    def __init__(self,name='Resource'):
        
        manager   = mp.Manager()
        
        elements  = manager.dict()
        checkinbox  = ShareableQueue(manager=manager)
        checkoutbox = ShareableQueue(manager=manager)        
        
        self.name        = name
        self.checkinbox  = checkinbox
        self.checkoutbox = checkoutbox
        self.manager     = manager
        
        self.out_service = None
        self.in_service  = None
        
    def start(self):
        if not isinstance(self.out_service,Service):
            raise AttributeError , 'no out_service!'
        if not isinstance(self.in_service,Service):
            raise AttributeError , 'no in_service!'
        self.out_service.start()
        self.in_service.start()
        
        return
    
    def gate(self,default=None):
        """ return remote for resource
        """
        checkout_remote = self.out_service.remote()
        checkin_remote  = self.in_service.remote()
        return ResourceGate(checkout_remote,checkin_remote,default)

        
# ----------------------------------------------------------------------
#   Resource Gate
# ----------------------------------------------------------------------

class ResourceGate(object):
    
    def __init__(self,checkout_remote,checkin_remote,default=None):
        
        self._checkout_remote = checkout_remote
        self._checkin_remote  = checkin_remote
        self.default         = default
        
    def __enter__(self,request=None):
        
        request = request or self.default
        if request is None:
            raise ResourceException , 'resource request not specified'
        
        self.request = request
        
        self.check_out(request)
        
    def __exit__(self,t,v,b):

        request = self.request
        
        self.check_in(request)
        
    def check_out(self,request):
        self._checkout_remote(request)
    
    def check_in(self,request):
        self._checkin_remote(request)


        

