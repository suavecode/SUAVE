
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import multiprocessing as mp
from GlobalManager import GlobalManager

# ----------------------------------------------------------------------
#   Sharable Queue
# ----------------------------------------------------------------------

def ShareableQueue(maxsize=None,manager=None):
    
    if manager is None:
        manager = GlobalManager()
    
    queue = manager.JoinableQueue(maxsize)
    return queue


# ----------------------------------------------------------------------
#   Tests
# ----------------------------------------------------------------------

if __name__ == '__main__':
    
    q1 = ShareableQueue(2)
    q2 = ShareableQueue(2)
    
    q1.put('a')
    q1.put('b')
    
    q2.put('c')
    
    print q1.get()
    print q1.get()
    print q2.get()
    #print q2.get(block=False) #should raise exception