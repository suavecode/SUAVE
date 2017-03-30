'''
Define the variables class

Rick Fenrich 2/14/17
'''

import numpy as np

class Variables():

    def __init__(self,role):
        
        self.role = role # design or parameter
        self.type = list()
        self.info = list()
        self.name = list()
        self.n = 0
        self.value = np.array([])
        
        if( self.role == 'design' ):
            self.lower_bound = np.array([])
            self.upper_bound = np.array([])
                    
    def add_variable(self,variable_type,**kwargs):
    
        # Assign attributes specific to each variable type
    	if( variable_type == 'continuous' ):
    	
    	    if( type(kwargs['value']) is np.ndarray ):
    	        self.value = np.hstack((self.value,np.squeeze(kwargs['value'])))
    	        s = np.squeeze(kwargs['value']).size	        
    	    elif( type(kwargs['value']) is list ):
    	        self.value = np.hstack((self.value,np.array(kwargs['value'])))
    	        s = len(kwargs['value']) 	        
    	    elif( type(kwargs['value']) is float or type(kwargs['value']) is int ):
    	        self.value = np.hstack((self.value,float(kwargs['value'])))
    	        s = 1
    	    else:
    	        raise NotImplementedError('%s not supported for variable type' % type(kwargs['value']))
    	    
    	    self.n = self.n + s
    	    self.type = self.type + ['continuous']*s
    	    self.info = self.info + [0]*s
    	    
    	elif( variable_type == 'lognormal' ):
    	
    	    if( type(kwargs['mean']) is np.ndarray ):
    	        mean = list(np.squeeze(kwargs['mean'])) # mean of underlying normal distribution
    	        variance = list(np.squeeze(kwargs['var'])) # variance of underlying normal distribution
    	    elif( type(kwargs['mean']) is list ):
    	        mean = kwargs['mean']
    	        variance = kwargs['var']
    	    elif( type(kwargs['mean']) is float or type(kwargs['mean']) is int ):
    	        mean = [float(kwargs['mean'])]
    	        variance = [float(kwargs['var'])]
    	    else:
    	        raise NotImplementedError('%s not supported for variable type' % type(kwargs['mean']))
    	    
    	    s = len(mean)
    	    self.n = self.n + s
    	    self.type = self.type + ['lognormal']*s
    	    for i in range(s):
    	        self.info.append([mean[i],variance[i]])
    	        
    	    if( 'value' in kwargs ): # set value to mean of random variable
                if( type(kwargs['value']) is np.ndarray ):
                    self.value = np.hstack((self.value,np.squeeze(kwargs['value'])))
                elif( type(kwargs['value']) is list ):
                    self.value = np.hstack((self.value,np.array(kwargs['value'])))
                elif( type(kwargs['value']) is float or type(kwargs['value']) is int ):
                    self.value = np.hstack((self.value,float(kwargs['value'])))
                else:
                    raise NotImplementedError('%s not supported for variable type' % type(kwargs['value']))
    	    else: # set value to mean of random variable
    	        value = np.zeros(s)
    	        for i in range(s):
    	            value[i] = np.exp(mean[i] + variance[i]/2) # evaluate at mean
    	        self.value = np.hstack((self.value,value))
    	
    	elif( variable_type == 'uniform' ):
    	
    	    if( type(kwargs['lower_bound']) is np.ndarray ):
    	        lb = list(np.squeeze(kwargs['lower_bound']))
    	        ub = list(np.squeeze(kwargs['upper_bound']))
    	    elif( type(kwargs['lower_bound']) is list ):
    	        lb = kwargs['lower_bound']
    	        ub = kwargs['upper_bound']
    	    elif( type(kwargs['lower_bound']) is float or type(kwargs['lower_bound']) is int ):
    	        lb = [float(kwargs['lower_bound'])]
    	        ub = [float(kwargs['upper_bound'])]
    	    else:
    	        raise NotImplementedError('%s not supported for variable type' % type(kwargs['lower_bound']))
    	
    	    s = len(lb)
    	    self.n = self.n + s
    	    self.type = self.type + ['uniform']*s
    	    for i in range(s):
    	        self.info.append([lb[i],ub[i]])
    	        
    	    if( 'value' in kwargs ): # set value to mean of random variable
                if( type(kwargs['value']) is np.ndarray ):
                    self.value = np.hstack((self.value,np.squeeze(kwargs['value'])))
                elif( type(kwargs['value']) is list ):
                    self.value = np.hstack((self.value,np.array(kwargs['value'])))
                elif( type(kwargs['value']) is float or type(kwargs['value']) is int ):
                    self.value = np.hstack((self.value,float(kwargs['value'])))
                else:
                    raise NotImplementedError('%s not supported for variable type' % type(kwargs['value']))
    	    else: # set value to mean of random variable
    	        value = np.zeros(s)
    	        for i in range(s):
    	            value[i] = (lb[i]+ub[i])/2 # evaluate at mean
    	        self.value = np.hstack((self.value,value))
    	        
    	
    	else:
            raise NotImplementedError('Variable type %s not implemented.' % variable_type)
    	
    	# Assign general attributes here
    	if( 'name' in kwargs ):
    	    if( type(kwargs['name']) is list ):
    	        self.name = self.name + kwargs['name']
    	    else:
    	        self.name.append(kwargs['name'])
    	else:
    	    self.name = self.name + ['no-name']*s
    	    
    	# Assign attributes specific to design variables here
    	if( self.role == 'design' ):
    	    if( type(kwargs['lower_bound']) is np.ndarray ):
    	        self.lower_bound = np.hstack((self.lower_bound,np.squeeze(kwargs['lower_bound'])))
    	        self.upper_bound = np.hstack((self.upper_bound,np.squeeze(kwargs['upper_bound'])))
    	    elif( type(kwargs['lower_bound']) is list ):
    	        self.lower_bound = np.hstack((self.lower_bound,np.array(kwargs['lower_bound'])))
    	        self.upper_bound = np.hstack((self.upper_bound,np.array(kwargs['upper_bound'])))
    	    elif( type(kwargs['lower_bound']) is float ):
    	        self.lower_bound = np.hstack((self.lower_bound,kwargs['lower_bound']))
    	        self.upper_bound = np.hstack((self.upper_bound,kwargs['upper_bound']))
    	    else:
    	        raise NotImplementedError('%s not supported for variable type' % type(kwargs['lower_bound']))
        elif( self.role == 'parameter' ):
            pass
        else:
            raise NotImplementedError('Variable role %s not implemented.' % self.role)
                
    def __print__(self):
    
        string = ''
    
        for i in range(self.n):
        
            t = self.type[i]
        
            if( t == 'continuous' ):
                if( self.role == 'design' ):
                    string += '%s: %s, value: %f, lb: %f, ub: %f\n' % (self.name[i],self.type[i],self.value[i],self.lower_bound[i],self.upper_bound[i])
                else:
                    string += '%s: %s, value: %f\n' % (self.name[i],self.type[i],self.value[i])
                #string += 'values:       {}\n'.format(" ".join(str(e) for e in self.value[i]))        
            elif( t == 'lognormal' ):
                string += '%s: %s, value: %f, mean: %f, variance %f\n' % (self.name[i],self.type[i],self.value[i],self.info[i][0],self.info[i][1])
                #string += 'values:    {}\n'.format(" ".join(str(e) for e in self.value))
                #string += 'means:     {}\n'.format(" ".join(str(e) for e in self.mean))
                #string += 'variances: {}\n'.format(" ".join(str(e) for e in self.variance))       
            elif( t == 'uniform' ):
                string += '%s: %s, value: %f, lb: %f, ub %f\n' % (self.name[i],self.type[i],self.value[i],self.info[i][0],self.info[i][1]) 
            else:
                raise NotImplementedError('Variable type %s not implemented for print.' % self.type)          
            
        return string
