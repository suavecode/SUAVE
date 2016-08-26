%module Turbofan_TASOPTc

%{
    
#define SWIG_FILE_WITH_INIT

#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <armadillo>
#include "Turbofan_TASOPTc.hpp"

%}

%include "numpy.i"
%include "std_string.i"
%include "std_map.i"
%include "std_vector.i"

%include <typemaps.i>
%init %{ import_array(); %}


//%include "/home/anilvariyar/Desktop/armanpy-0.1.4/include/armanpy.i"
%include "Turbofan_TASOPTc.hpp"

