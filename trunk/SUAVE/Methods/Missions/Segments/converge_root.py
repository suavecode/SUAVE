## @ingroup Methods-Missions-Segments
# converge_root.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import scipy.optimize
import numpy as np

from SUAVE.Core.Arrays import array_type

# ----------------------------------------------------------------------
#  Converge Root
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments
def converge_root(segment):
    """Interfaces the mission to a numerical solver. The solver may be changed by using root_finder.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    segment                            [Data]
    segment.settings.root_finder       [Data]
    state.numerics.tolerance_solution  [Unitless]

    Outputs:
    state.unknowns                     [Any]
    segment.state.numerics.converged   [Unitless]

    Properties Used:
    N/A
    """      
    
    ## ----------------------------------------------------------------------------------------------
    ## minimize 
    
    #unknowns = segment.state.unknowns.pack_array()
    
    #try:
        #root_finder = segment.settings.root_finder
    #except AttributeError:
        #root_finder = scipy.optimize.minimize     
        
    #def minimize_iterate(unknowns, segment):
        #residuals = iterate(unknowns, segment)
        #ret = scipy.linalg.norm(residuals)
        #print(ret)
        #return ret
    
    #res = root_finder( minimize_iterate,
                    #unknowns,
                    #args = segment,
                    #tol = segment.state.numerics.tolerance_solution,
                    #method = 'Powell')
    
    #raise NotImplementedError
    
    #if ier!=1:
        #print("Segment did not converge. Segment Tag: " + segment.tag)
        #print("Error Message:\n" + msg)
        #segment.state.numerics.converged = False
        #segment.converged = False
    #else:
        #segment.state.numerics.converged = True
        #segment.converged = True
                            
    #return
    ## ----------------------------------------------------------------------------------------------    
    
    # ----------------------------------------------------------------------------------------------
    # fsolve 
    
    unknowns = segment.state.unknowns.pack_array()
    
    try:
        root_finder = segment.settings.root_finder
    except AttributeError:
        root_finder = scipy.optimize.fsolve     
        
    def add_unknowns(body_a, wind_a, pos):
        a = np.array([5.23598776e-02, 5.23598776e-02, 5.23598776e-02, 5.23598776e-02, 1.74532925e-02, 1.74532925e-02, 1.74532925e-02, 1.74532925e-02, 8.00000000e+01])
        a[pos] = body_a*np.pi/180
        a[pos+4] = wind_a*np.pi/180
        return a
    
    unknowns,infodict,ier,msg = root_finder( iterate,
                                         unknowns,
                                         args = segment,
                                         xtol = segment.state.numerics.tolerance_solution,
                                         maxfev = segment.state.numerics.max_evaluations,
                                         full_output = 1)
    
    if ier!=1:
        print("Segment did not converge. Segment Tag: " + segment.tag)
        print("Error Message:\n" + msg)
        segment.state.numerics.converged = False
        segment.converged = False
    else:
        segment.state.numerics.converged = True
        segment.converged = True
                            
    return
    # ----------------------------------------------------------------------------------------------
    
    # ----------------------------------------------------------------------------------------------
    # root 
    
    #state = segment.state
    #unknowns = state.unknowns.pack_array()
    
    #try:
        #root_finder = segment.settings.root_finder
    #except AttributeError:
        ##root_finder = scipy.optimize.fsolve
        #root_finder = scipy.optimize.root
    
    #sol = root_finder( iterate,
                                         #unknowns,
                                         #args = segment,
                                         #tol = state.numerics.tolerance_solution,
                                         #method='lm')
    
    #unknowns = sol.x
    #print("Status: " + str(sol.success) + " Segment: " + segment.tag + " Message: " + sol.message)
    
    ##if sol.status!=1:
        ##print "Segment did not converge. Segment Tag: " + segment.tag
        ##print "Error Message:\n" + sol.message
        ##segment.state.numerics.converged = False
    ##else:
        ##segment.state.numerics.converged = True    

    ##if ier!=1:
        ##print "Segment did not converge. Segment Tag: " + segment.tag
        ##print "Error Message:\n" + msg
        ##segment.state.numerics.converged = False
    ##else:
        ##segment.state.numerics.converged = True
         
                            
    #return    
    
    # ----------------------------------------------------------------------------------------------
    
# ----------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments
def iterate(unknowns, segment):
    
    """Runs one iteration of of all analyses for the mission.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    state.unknowns                [Data]
    segment.process.iterate       [Data]

    Outputs:
    residuals                     [Unitless]

    Properties Used:
    N/A
    """       
    if isinstance(unknowns,array_type):
        segment.state.unknowns.unpack_array(unknowns)
    else:
        segment.state.unknowns = unknowns
        
    segment.process.iterate(segment)
    
    residuals = segment.state.residuals.pack_array()
        
    return residuals 