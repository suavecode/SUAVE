
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from OrderedBunch import OrderedBunch
from IndexableDict import IndexableDict


# ----------------------------------------------------------------------
#   Indexable Bunch
# ----------------------------------------------------------------------

class IndexableBunch(IndexableDict,OrderedBunch):
    """ An ordered indexable dictionary that provides attribute-style access.
    """
    pass
    # ballin

# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------

if __name__ == '__main__':
    
    o = IndexableBunch()
    o['x'] = 'hello'
    o.y = 1
    o['z'] = [3,4,5]
    o.t = IndexableBunch()
    o.t['h'] = 20
    o.t.i = (1,2,3)
    
    print o

    import pickle

    d = pickle.dumps(o)
    p = pickle.loads(d)
    
    print ''
    print p    
    
    o.t[0] = 'changed'
    p.update(o)

    print ''
    print p
    print p[0]
    