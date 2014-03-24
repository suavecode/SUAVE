# test_Data.py

import SUAVE
from SUAVE.Structure import Data
from copy import deepcopy

# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

def main():
    
    test_data()
    
    return


# ----------------------------------------------------------------------
#  Sub-Classing DataBase
# ----------------------------------------------------------------------

class MyData(Data):    
    """ description of database
    """
    
    # default/initial values for database items, 
    # also documents the contents of the class
    def __defaults__(self):
        self.key1 = 1
        self.key2 = [1,2,3]
        self.key3 = 'hilo'
    
#: class MyData()


# ----------------------------------------------------------------------
#  Testing Database
# ----------------------------------------------------------------------

def test_data():
    
    # the SUAVE Data class...
    data = Data()
    
    # item-style assignment
    data['key1'] = 'iterm1'
    data['key2'] = 'borkaloo'
    
    # attribute-style assignment
    data.key3 = ['magic','majic','majik']
    data.key4 = MyData()
    
    # adding a branch
    data.key5 = Data()
    data.key5.key_a = 2
    
    # index based access
    print data[0]
    print data.values()[0] # an equivalent syntax
    
    print ""
    
    # iteration, equivalent syntax
    for i in range( len(data) ):
        print data[i]
    for value in data.values():
        print value
    for i,value in enumerate( data.values() ):
        print value
        
    print ""
        
    # very pythonic key-value iteration (!!!)
    for key,value in data.iteritems():
        print key,": ",value
    
    print ""
        
    # check if data has a key
    if data.has_key('key1'):
        print 'i haz key1!'
        
    print ""
    
    # check if data is a certain type
    if isinstance(data,MyData):
        print "this won't print"
    elif isinstance(data.key4,MyData):
        print 'data.key4 is of type MyData!'
    
    print ""
        
    # an example of deep vs reference copy
    refer = data
    deep  = deepcopy(data)
    
    print data.key1
    refer.key1 = 'new_value!!'
    print data.key1
    deep.key1 = "won't change data"
    print data.key1
    
    # the relevent case of the deepcopy example is when receiving input
    # to a method.  the input will be a reference copy, and if you don't 
    # want changes made to the data to propogate back to the referenced
    # object, you must deepcopy it.
    
    print ""
    
    return

#: def test_data()


# ----------------------------------------------------------------------
#  Call Main
# ----------------------------------------------------------------------

if __name__ == '__main__':
    main()
