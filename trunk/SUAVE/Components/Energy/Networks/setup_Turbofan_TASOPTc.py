from distutils.core import setup, Extension

import numpy

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


code_module =Extension('_Turbofan_TASOPTc',

sources=['Turbofan_TASOPTc_wrap.cxx','Turbofan_TASOPTc.cpp'],
                       include_dirs = [numpy_include])

setup(name        ='Turbofan_TASOPTc',
      ext_modules  =[code_module])



