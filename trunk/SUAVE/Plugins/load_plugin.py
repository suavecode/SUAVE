## @ingroup Plugins
import os, sys
__dir__ = os.path.abspath(os.path.dirname(__file__))

## @ingroup Plugins
def load_plugin(package_name):
    """This function loads a package that uses absolute package imports
    by temporarily modifying the python sys.path.

    Assumptions:
    Packages are in the same folder as load_plugin.py

    Source:
    N/A

    Inputs:
    package_name   <string> name of the package to import

    Outputs:
    package        imported package

    Properties Used:
    N/A
    """     

    # save current path
    saved_path = sys.path
    
    # remove references to package name in path
    paths = [ p  for p in sys.path  if package_name.lower() in p.lower() ]
    
    for p in paths:  sys.path.remove(p)
    
    # add the plugin path
    sys.path.append( __dir__ )
    
    package = __import__(package_name)
    #package = reload(package)
    
    sys.path = saved_path
    
    return package


