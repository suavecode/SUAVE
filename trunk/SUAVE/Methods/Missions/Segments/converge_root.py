# converge_root.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# Scipy
import scipy
import scipy.optimize

from SUAVE.Core.Arrays import array_type
from autograd.numpy import np
from autograd.numpy.numpy_extra import ArrayNode
from autograd import jacobian, grad, hessian

from SUAVE.Core.Deep_Core import OrderedDict

# ----------------------------------------------------------------------
#  Converge Root
# ----------------------------------------------------------------------

def converge_root(segment,state):
    
    unknowns = state.unknowns.pack_array()
    
    #unknowns = np.ravel(np.transpose(np.reshape(unknowns, (2,-1))))
    
    ua = np.array([])
    ub = np.array([])
    
    for ii in xrange(0,len(unknowns)/16,2):
        ua = np.append(ua,unknowns[16*ii:16*(ii+1)])
        ub = np.append(ub,unknowns[16*ii+16:16*(ii+1)+16])
        
    unknowns = np.ravel(np.vstack((ua,ub)).T)
        
        
    
    try:
        root_finder = segment.settings.root_finder
    except AttributeError:
        root_finder = scipy.optimize.fsolve 
    
    prime  = jacobian(iterate)
    prime2 = hessian(iterate)
    
    #prime(unknowns,(segment,state))
    
    
    #unknowns = root_finder( iterate,
                            #unknowns,
                            #args = [segment,state],
                            #xtol = state.numerics.tolerance_solution,
                            #fprime = prime)
                            
    unknowns = scipy.optimize.root(iterate,unknowns,args = [segment,state],method='krylov',jac=prime)
    
    #unknowns = scipy.optimize.newton(iterate,unknowns,args = [segment,state])
    
    #unknowns
    #res = scipy.optimize.minimize(iterate, unknowns, args = (segment,state),tol=1e-9,method='BFGS',jac=prime,hess=prime2)
    #unknowns = res.x
    
    #unknowns = scipy.optimize.leastsq(iterate,unknowns,args = (segment,state),xtol=state.numerics.tolerance_solution,Dfun=prime)

    return
    
# ----------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------

def iterate(unknowns,segment,state):
        
    state.unknowns = unpack_autograd(state.unknowns, unknowns)
    #state.unknowns.unpack_array(unknowns)
    segment.process.iterate(segment,state)
    #residuals = np.reshape(state.residuals.forces[:,:],(len(unknowns)))
    
    residuals = pack_autograd(state.residuals)
    
    #residuals = np.power(np.dot(residuals,residuals),0.5)
    #print residuals2
    
    #residuals = state.residuals.pack_array()
        
    return residuals 


#def unpack_autograd(s_unkowns,unknowns):
    
    ## We need to take the grad object and slice it into the dictionary   
    
    #unpack_autograd.count = 0
    
    #def unpack(dic,unknowns):
        
        #for key,val in dic.iteritems():
            #if isinstance(dic[key],OrderedDict):
                #unpack(dic[key], unknowns)
                #continue
            #elif isinstance(dic[key],str):continue
            
            #n = len(val)
            #c = unpack_autograd.count
            #dic[key] = np.reshape(unknowns[c:(n+c)],np.shape(val))
            #unpack_autograd.count = c + n
            
        #return s_unkowns

    
    #s_unkowns = unpack(s_unkowns, unknowns)

        
    #return s_unkowns
    

def unpack_autograd(s_unkowns,unknowns):
    
    # We need to take the grad object and slice it into the dictionary   
    
    unpack_autograd.count = 0
    unpack_autograd.last = 0
    
    def unpack(dic,unknowns):
        
        for key,val in dic.iteritems():
            if isinstance(dic[key],OrderedDict):
                unpack(dic[key], unknowns)
                continue
            elif isinstance(dic[key],str):continue
            
            n    = len(val)
            c    = unpack_autograd.count
            last = unpack_autograd.last
            dic[key] = np.reshape(unknowns[last:last+n*2:2],np.shape(val))
            last = last + (2*n*c) -1*c + (1+c)%2
            
            c    = (c + 1)%2
            unpack_autograd.last  = last
            unpack_autograd.count = c
            
        return s_unkowns

    
    s_unkowns = unpack(s_unkowns, unknowns)

        
    return s_unkowns


def pack_autograd(s_residuals):
    
    # We are going to loop through the dictionary recursively and unpack
    
    dic = s_residuals
    pack_autograd.array = np.array([])
    
    def pack(dic):
        for key in dic.iterkeys():
            if isinstance(dic[key],OrderedDict):
                pack(dic[key]) # Regression
                continue
            elif np.rank(dic[key])>2: continue
            elif isinstance(dic[key],str):continue
            
            #pack_autograd.array = np.append(pack_autograd.array,np.transpose(dic[key]))
            pack_autograd.array = np.append(pack_autograd.array,dic[key])
            
            
    pack(dic)
    residuals = pack_autograd.array 

        
    return residuals