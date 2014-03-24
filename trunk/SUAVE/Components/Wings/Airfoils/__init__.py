
# classes
from Airfoil import Airfoil

# packages
# ...

# functions
from load_airfoils import load_airfoils

# load airfoils
import os
__dir__ = os.path.split(__file__)[0]
_airfoils = load_airfoils(__dir__)
for k,v in _airfoils.items():
    exec('%s = v'%k)
#del os, _airfoils, k, v
