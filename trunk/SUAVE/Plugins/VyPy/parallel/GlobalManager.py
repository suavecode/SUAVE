
import multiprocessing as mp

global manager
manager = None

def GlobalManager():    
    
    # global mananger (local to machine) for sharable queues
    global manager
    
    if manager is None:
        manager = mp.Manager()
    
    return manager
    