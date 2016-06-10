## Data.py
##
## Created:  Jan 2015, T. Lukacyzk
## Modified: Feb 2016, T. MacDonald

#""" SUAVE Data Base Classes
#"""

#from Data_base import Data_Base

## for enforcing attribute style access names
#import string
#chars = string.punctuation + string.whitespace
#t_table = string.maketrans( chars          + string.uppercase , 
                            #'_'*len(chars) + string.lowercase )

##from warnings import warn

## ----------------------------------------------------------------------
##   Data
## ----------------------------------------------------------------------        

#class Data(Data_Base):
    
    #def __defaults__(self):
        #return 
    
    #def __repr__(self):
        #return self.dataname()    
    
    #def append(self,value,key=None):
        #if key is None: key = value.tag
        #key_in = key
        #key = key.translate(t_table)
        #if key != key_in: warn("changing appended key '%s' to '%s'\n" % (key_in,key))
        #DataBunch.append(self,value,key)
        
        
    #def pack_array(self,output='vector'):
        #""" OrderedDict.pack_array(output='vector')
            #maps the data dict to a 1D vector or 2D column array
            
            #Inputs - 
                #output - either 'vector' (default), or 'array'
                         #chooses whether to output a 1d vector or 
                         #2d column array
            #Outputs - 
                #array - the packed array
                
            #Assumptions - 
                #will only pack int, float, np.array and np.matrix (max rank 2)
                #if using output = 'matrix', all data values must have 
                #same length (if 1D) or number of rows (if 2D), otherwise is skipped
        
        #"""
        
        ## dont require dict to have numpy
        #import numpy as np
        #from Deep_Core.arrays import atleast_2d_col, array_type, matrix_type
        
        ## check output type
        #if not output in ('vector','array'): raise Exception , 'output type must be "vector" or "array"'        
        #vector = output == 'vector'
        
        ## list to pre-dump array elements
        #M = []
        
        ## valid types for output
        #valid_types = ( int, float,
                        #array_type,
                        #matrix_type )
        
        ## initialize array row size (for array output)
        #size = [False]
        
        ## the packing function
        #def do_pack(D):
            #for v in D.itervalues():
                ## type checking
                #if isinstance( v, OrderedDict ): 
                    #do_pack(v) # recursion!
                    #continue
                #elif not isinstance( v, valid_types ): continue
                #elif np.rank(v) > 2: continue
                ## make column vectors
                #v = atleast_2d_col(v)
                ## handle output type
                #if vector:
                    ## unravel into 1d vector
                    #v = v.ravel(order='F')
                #else:
                    ## check array size
                    #size[0] = size[0] or v.shape[0] # updates size once on first array
                    #if v.shape[0] != size[0]: 
                        ##warn ('array size mismatch, skipping. all values in data must have same number of rows for array packing',RuntimeWarning)
                        #continue
                ## dump to list
                #M.append(v)
            ##: for each value
        
        ## do the packing
        #do_pack(self)
        
        ## pack into final array
        #if M:
            #M = np.hstack(M)
        #else:
            ## empty result
            #if vector:
                #M = np.array([])
            #else:
                #M = np.array([[]])
        
        ## done!
        #return M
    
    #def typestring(self):
        ## build typestring
        #typestring = str(type(self)).split("'")[1]
        #typestring = typestring.split('.')
        #if typestring[-1] == typestring[-2]:
            #del typestring[-1]
        #typestring = '.'.join(typestring) 
        #return typestring
    
    #def dataname(self):
        #return "<data object '" + self.typestring() + "'>"
    
    
    #def unpack_array(self,M):
        #""" OrderedDict.unpack_array(array)
            #unpacks an input 1d vector or 2d column array into the data dictionary
                #following the same order that it was unpacked
            #important that the structure of the data dictionary, and the shapes
                #of the contained values are the same as the data from which the 
                #array was packed
        
            #Inputs:
                 #array - either a 1D vector or 2D column array
                 
            #Outputs:
                 #a reference to self, updates self in place
                 
        #"""
        
        ## dont require dict to have numpy
        #import numpy as np
        #from Deep_Core.arrays import atleast_2d_col, array_type, matrix_type
        
        ## check input type
        #vector = np.rank(M) == 1
        
        ## valid types for output
        #valid_types = ( int, float,
                        #array_type,
                        #matrix_type )
        
        ## counter for unpacking
        #_index = [0]
        
        ## the unpacking function
        #def do_unpack(D):
            #for k,v in D.iteritems():
                
                ## type checking
                #if isinstance(v,OrderedDict): 
                    #do_unpack(v) # recursion!
                    #continue
                #elif not isinstance(v,valid_types): continue
                
                ## get this value's rank
                #rank = np.rank(v)
                
                ## get unpack index
                #index = _index[0]                
                
                ## skip if too big
                #if rank > 2: 
                    #continue
                
                ## scalars
                #elif rank == 0:
                    #if vector:
                        #D[k] = M[index]
                        #index += 1
                    #else:#array
                        #continue
                        ##raise RuntimeError , 'array size mismatch, all values in data must have same number of rows for array unpacking'
                    
                ## 1d vectors
                #elif rank == 1:
                    #n = len(v)
                    #if vector:
                        #D[k][:] = M[index:(index+n)]
                        #index += n
                    #else:#array
                        #D[k][:] = M[:,index]
                        #index += 1
                    
                ## 2d arrays
                #elif rank == 2:
                    #n,m = v.shape
                    #if vector:
                        #D[k][:,:] = np.reshape( M[index:(index+(n*m))] ,[n,m], order='F')
                        #index += n*m 
                    #else:#array
                        #D[k][:,:] = M[:,index:(index+m)]
                        #index += m
                
                ##: switch rank
                
                #_index[0] = index

            ##: for each itme
        
        ## do the unpack
        #do_unpack(self)
         
        ## check
        #if not M.shape[-1] == _index[0]: warn('did not unpack all values',RuntimeWarning)
         
        ## done!
        #return self    
    
    
    #def deep_set(self,keys,val):
        
        #if isinstance(keys,str):
            #keys = keys.split('.')
        
        #data = self
         
        #if len(keys) > 1:
            #for k in keys[:-1]:
                #data = data[k]
        
        #data[ keys[-1] ] = val
        
        #return data

    #def deep_get(self,keys):
        
        #if isinstance(keys,str):
            #keys = keys.split('.')
        
        #data = self
         
        #if len(keys) > 1:
            #for k in keys[:-1]:
                #data = data[k]
        
        #value = data[ keys[-1] ]
        
        #return value    


## ----------------------------------------------------------------------
##   Module Tests
## ----------------------------------------------------------------------        

#if __name__ == '__main__':
    
    #d = Data()
    #d.tag = 'data name'
    #d['value'] = 132
    #d.options = Data()
    #d.options.field = 'of greens'
    #d.options.half  = 0.5
    #print d
    
    #import numpy as np
    #ones = np.ones([10,1])
        
    #m = Data()
    #m.tag = 'numerical data'
    #m.hieght = ones * 1.
    #m.rates = Data()
    #m.rates.angle  = ones * 3.14
    #m.rates.slope  = ones * 20.
    #m.rates.special = 'nope'
    #m.value = 1.0
    
    #print m
    
    #V = m.pack_array('vector')
    #M = m.pack_array('array')
    
    #print V
    #print M
    
    #V = V*10
    #M = M-10
    
    #print m.unpack_array(V)
    #print m.unpack_array(M)
    
