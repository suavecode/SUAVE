
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import os, shutil, sys, copy
import numpy as np

from VyPy.tools.arrays import array_type, matrix_type, atleast_2d

from VyPy.data import OrderedDict, OrderedBunch
odict  = OrderedDict
obunch = OrderedBunch

from VyPy.data import IndexableDict, IndexableBunch
idict  = IndexableDict
ibunch = IndexableBunch

iterable_type = (list,tuple,array_type,matrix_type)

from VyPy.data import scaling


# ----------------------------------------------------------------------
#   Variable Object
# ----------------------------------------------------------------------

class Variable(object):
    
    Container = None
    
    def __init__( self, tag='x%i', initial=0.0,
                  bounds=(1.e-100,1.e+100), scale=1.0 ):
        
        self.tag     = tag
        self.initial = initial
        self.bounds  = bounds
        self.scale   = scale
        self.scaled  = None
    
    def __repr__(self):
        return '<Variable %s>' % self.tag


# ----------------------------------------------------------------------
#   Scaled Variable Object
# ----------------------------------------------------------------------

class ScaledVariable(object):

    Container = None
    
    def __init__( self, tag, initial=0.0,
                  bounds=(1.e-100,1.e+100), scale=1.0 ):
        
        self.tag     = tag
        self.initial = initial/scale
        self.bounds  = tuple([ b/scale for b in bounds ])
        self.scale   = scale
    
    def __repr__(self):
        return '<ScaledVariable %s>' % self.tag


# ----------------------------------------------------------------------
#   Variables Container
# ----------------------------------------------------------------------

class Variables(IndexableDict):
    
    def __init__(self):
        self.scaled = ScaledVariables(self)
    
    def __set__(self,problem,arg_list):
        self.clear()
        self.extend(arg_list)
        
    def extend(self,arg_list):
        for args in arg_list:
            self.append(*args)

    def append( self, tag, initial=0.0,
                bounds=(1.e-100,1.e+100), scale=1.0 ):

        if isinstance(tag,Variable):
            variable = tag
            tag     = variable.tag
            initial = variable.initial
            bounds  = variable.bounds
            scale   = variable.scale
        else:
            tag = self.next_key(tag)
            variable = Variable(tag,initial,bounds,scale)
            
        if variable.scale == 'bounds':
            scale = scaling.Linear( bounds[1]-bounds[0] , (bounds[1]+bounds[0])/2 )
            variable.scale = scale
            
        scaled_var = ScaledVariable(tag,initial,bounds,scale)
        variable.scaled = scaled_var
        
        if self.has_key(tag):
            print 'Warning: overwriting variable %s' % tag
            
        self[tag]        = variable
        self.scaled[tag] = scaled_var
        
        return
    
    def unpack_array(self,values):
        """ vars = Variables.unpack_array(vals)
            unpack a list of values into an ordered dictionary 
            of variables
        """
        
        values = np.ravel(values)
        variables = ibunch(zip(self.keys(),self.initials()))
        variables.unpack_array(values)
        
        return variables
    
    def pack_array(self,variables):
        """ values = Variables.pack_array(vars)
            pack an ordered dictionary of variables into
            a 1D array of values
        """
        
        if isinstance(variables,odict):
            # pack Variables
            values = variables.pack_array('vector')
        elif isinstance(variables,iterable_type):
            # already packed
            values = np.ravel(variables)  
        else:
            raise Exception, 'could not pack variables: %s' % variables
                
        return values
    
    def tags(self):
        return self.keys()
    def initials(self):
        return [ var.initial for var in self.values() ]
    def bounds(self):
        return [ var.bounds for var in self.values() ]
    def scales(self):
        return [ var.scale for var in self.values() ]
    
    def initials_array(self):
        return np.vstack([ atleast_2d(x,'col') for x in self.initials() ])
    def bounds_array(self):
        return np.vstack([ atleast_2d(b,'col').T for b in self.bounds() ])
    
    def set(self,initials=None,bounds=None,scales=None):
        if initials:
            for i,(v,s) in enumerate(zip(initials,self.scales())):
                self[i].initial = v
                self.scaled[i].initial = v/s
        if scales:
            for i,s in enumerate(scales):
                self[i].scale = s
                self[i].scaled.scale = s
        if bounds:
            for i,(bnd,s) in enumerate(zip(bounds,self.scales())):
                self[i].bounds = b
                self[i].scaled.bounds = [ v/s for v in b ]
    

# ----------------------------------------------------------------------
#   Scaled Variables Container
# ----------------------------------------------------------------------   

class ScaledVariables(Variables):
        
    def __init__(self,variables):
        self.unscaled = variables
     
    def unpack_array(self,values):
        """ vars = Variables.unpack_array(vals)
            unpack_array an array of scaled values into
            an ordered dictionary of unscaled variables
        """
        scales = self.scales()
        variables = Variables.unpack_array(self,values)
        for k,s in zip (variables.keys(),scales):
            variables[k] = variables[k] * s
        return variables
    
    def pack_array(self,variables):
        """ values = Variables.pack_array(vars)
            pack an ordered dictionary of unscaled variables into
            an array of scaled values
        """
        scales = self.scales()
        for k,s in zip (variables.keys(),scales):
            variables[k] = variables[k] / s
        values = Variables.pack_array(self,variables)
        return values
    
    def __set__(self,*args):
        ''' not used '''
        raise AttributeError('__set__')
    def append(self,*args):
        ''' not used '''
        raise AttributeError('append')
    def extend(self,*args):
        ''' not used '''
        raise AttributeError('extend')
    
    def set(self,initials=None,bounds=None,scales=None):
        if initials:
            for i,(v,s) in enumerate(zip(initials,self.scales)):
                self[i].initial = v
                self.unscaled[i].initial = v*s
        if bounds:
            for i,(bnd,s) in enumerate(zip(bounds,self.scales)):
                self[i].bounds = bnd
                self[i].unscaled.bounds = [ v*s for v in bnd ]
        if scales:
            for i,s in enumerate(scales):
                self[i].scale = s
                self[i].unscaled.scale = s
                
# ----------------------------------------------------------------------
#   Scaled Variables Container
# ----------------------------------------------------------------------   

Variable.Container       = Variables
ScaledVariable.Container = ScaledVariables