
import copy, pickle
from collections import OrderedDict
from Indexable_Bunch import Indexable_Bunch
from Data import Data

class Test(Data):
    def __defaults__(self):
        self.tag = 'Atmosphere'
        self.alt = 0.0      # m
        self.oat = [15.0]     # deg C
        self.nu  = 1.46e-5  # m^2/s
    
class MoreTest(Test):
    def __defaults__(self):
        self.tag = 'Watzup'
        self.alt   = 2.0
        self.hello = 'world'
        self.cool  = {'dict':'name','more':'stuff'}
        
    
T = Test()
T.f = 'new field'
print T

print "-------------- \n"

U = MoreTest(tag = 'new tag')
print T
print U

print "-------------- \n"

U.cool['dict'] = 'changed'
del U.hello

print T
print U

print "-------------- \n"

V = MoreTest()
V.tag = 'one more data'

print T
print U
print V

print "-------------- \n"

A = copy.deepcopy(U)
A.tag = 'copied'
b = pickle.dumps(A)
B = pickle.loads(b)
B.tag = 'pickled'

print T
print U
print V
print A
print B

print "-------------- \n"

B.insert('nu','zu',['inserted',703])
B.swap('alt','nu')

print T
print U
print V
print A
print B

print "-------------- \n"

L = B.linked_copy()
print L.tag
print (L.alt + L.oat[0])
Q = L.linked_copy()
print Q.tag

B.tag = 'this value is linked'
L.alt = 'only L and Q'
print L.tag
print Q.tag
print type(Q.tag)
print Q.get_bases()
print ""

del L.oat

print T
print U
print V
print A 
print B
print L 
print Q

