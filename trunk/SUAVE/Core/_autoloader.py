# imports
import os, shutil, glob

# file directory
__dir__ = os.path.split(__file__)[0]

# find all module files
modules = [ os.path.splitext(os.path.basename(f))[0] \
            for f in glob.glob(os.path.dirname(__file__)+"/*.py") \
            if not os.path.basename(f).startswith('_') ]

# find all packages
packages = [ d for d in os.listdir(__dir__) \
            if os.path.isdir(os.path.join(__dir__,d)) ]

# find all precompiled
#compiles = glob.glob(os.path.join(__dir__,'*.pyc'))

__all__ = modules + packages

# clean *.pyc
#c = ''
#for c in compiles:
    #os.remove(c)
#del c, compiles

# import modules
m=''
for m in modules:
    # print m
    #try:
    exec('from %s import %s'%(m,m))
    #except ImportError:
        #exec('import %s'%m)
del m, modules

# import packages
p=''
for p in packages:
    # print p , '/'
    exec('import %s'%p)
del p, packages

del os, shutil, glob