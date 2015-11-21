
import glob
import os

# new keys by wild card integer
def next_path(path_wild,format='%i'):
    """ finds the next index to use on a indexed key and applies it to key_wild
        key_wild is a string containing '*' to indicate where to increment
        the key
    """
    
    ksplit = key_wild.split('*')
    
    folders = glob.glob(path_wild)
    
    keys = []
    for f in folders:
        if not os.path.isdir(k): continue
        try:
            i = int( f.lstrip(ksplit[0]).rstrip(ksplit[1]) )
            keys.append(i)
        except:
            pass
        
    if keys:
        key_index = max(keys)+1
    else:
        key_index = 0
    
    path = path_wild.replace('*',format) % (key_index)
    
    return path