## @ingroup Methods-Missions-Segments
# optimize.py
# 
# Created:  Dec 2016, E. Botero
# Modified: Jun 2017, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import scipy.optimize as opt
import numpy as np

from SUAVE.Core.Arrays import array_type
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  Converge Root
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments
def converge_opt(segment,state):
    """Interfaces the mission to an optimization algorithm

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    state.unknowns                     [Data]
    segment                            [Data]
    state                              [Data]
    segment.algorithm                  [string]

    Outputs:
    state.unknowns                     [Any]

    Properties Used:
    N/A
    """     
    
    # pack up the array
    unknowns = state.unknowns.pack_array()
    
    # Have the optimizer call the wrapper
    obj       = lambda unknowns:get_objective(unknowns,(segment,state))   
    econ      = lambda unknowns:get_econstraints(unknowns,(segment,state)) 
    iecon     = lambda unknowns:get_ieconstraints(unknowns,(segment,state)) 
    
    # Setup the bnds of the problem
    bnds = make_bnds(unknowns, (segment,state))
    
    # Solve the problem, based on chosen algorithm
    if segment.algorithm == 'SLSQP':
        unknowns = opt.fmin_slsqp(obj,unknowns,f_eqcons=econ,f_ieqcons=iecon,bounds=bnds,iter=2000)
        
    elif segment.algorithm == 'SNOPT':
        
        # SNOPT imports
        import pyOpt
        import pyOpt.pySNOPT    
        
        # Have the optimizer call the wrapper
        obj_pyopt = lambda unknowns:get_problem_pyopt(unknowns,(segment,state)) 
    
        opt_prob = pyOpt.Optimization('SUAVE',obj_pyopt)
        opt_prob.addObj(segment.objective)
    
        for ii in xrange(0,len(unknowns)):
            lbd = (bnds[ii][0])
            ubd = (bnds[ii][1])
            vartype = 'c'
            opt_prob.addVar(str(ii),vartype,lower=lbd,upper=ubd,value=unknowns[ii])  
    
        # Setup constraints
        segment_points = state.numerics.number_control_points
        for ii in xrange(0,2*segment_points):
            opt_prob.addCon(str(ii), type='e', equal=0.)
        for ii in xrange(0,3*segment_points-1):
            opt_prob.addCon(str(ii+segment_points*2), type='i', lower=0.,upper=np.inf)
    
        print opt_prob
    
        snopt = pyOpt.pySNOPT.SNOPT()    
        outputs = snopt(opt_prob) 
    
        print outputs
        print opt_prob.solution(0)

    return
    
# ----------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------
    
## @ingroup Methods-Missions-Segments
def get_objective(unknowns,(segment,state)):
    """ Runs the mission if the objective value is needed
    
        Assumptions:
        N/A
        
        Inputs:
        state.unknowns      [Data]
    
        Outputs:
        objective           [float]

        Properties Used:
        N/A
                                
    """      
    
    if isinstance(unknowns,array_type):
        state.unknowns.unpack_array(unknowns)
    else:
        state.unknowns = unknowns
        
    if not np.all(state.inputs_last == state.unknowns):       
        segment.process.iterate(segment,state)
        
    objective = state.objective_value
    
    return objective

## @ingroup Methods-Missions-Segments
def get_econstraints(unknowns,(segment,state)):
    """ Runs the mission if the equality constraint values are needed
    
        Assumptions:
        N/A
        
        Inputs:
        state.unknowns      [Data]
            
        Outputs:
        constraints          [array]

        Properties Used:
        N/A
                                
    """       
    
    if isinstance(unknowns,array_type):
        state.unknowns.unpack_array(unknowns)
    else:
        state.unknowns = unknowns
        
    if not np.all(state.inputs_last == state.unknowns):       
        segment.process.iterate(segment,state)

    constraints = state.constraint_values
    
    return constraints

## @ingroup Methods-Missions-Segments
def make_bnds(unknowns,(segment,state)):
    """ Automatically sets the bounds of the optimization.
    
        Assumptions:
        Restricts throttle to between 0 and 100%
        Restricts body angle from 0 to pi/2 radians
        Restricts flight path angle from 0 to pi/2 radians
        
        Inputs:
        none
            
        Outputs:
        bnds

        Properties Used:
        N/A
                                
    """      
    
    ones    = state.ones_row(1)
    ones_m1 = state.ones_row_m1(1)
    ones_m2 = state.ones_row_m2(1)
    
    throttle_bnds = ones*(0.,1.)
    body_angle    = ones*(0., np.pi/2.)
    gamma         = ones*(0., np.pi/2.)
    
    if segment.air_speed_end is None:
        vels      = ones_m1*(0.,2000.)
    elif segment.air_speed_end is not None:    
        vels      = ones_m2*(0.,2000.)
    
    bnds = np.vstack([vels,throttle_bnds,gamma,body_angle])
    
    bnds = list(map(tuple, bnds))
    
    return bnds

## @ingroup Methods-Missions-Segments
def get_ieconstraints(unknowns,(segment,state)):
    """ Runs the mission if the inequality constraint values are needed
    
        Assumptions:
        Time only goes forward
        CL is less than a specified limit
        All altitudes are greater than zero
        
        Inputs:
        state.unknowns      [Data]
            
        Outputs:
        constraints          [array]

        Properties Used:
        N/A
                                
    """      
    
    if isinstance(unknowns,array_type):
        state.unknowns.unpack_array(unknowns)
    else:
        state.unknowns = unknowns
        
    if not np.all(state.inputs_last == state.unknowns):       
        segment.process.iterate(segment,state)
    
    # Time goes forward, not backward
    t_final = state.conditions.frames.inertial.time[-1,0]
    time_con = (state.conditions.frames.inertial.time[1:,0] - state.conditions.frames.inertial.time[0:-1,0])/t_final
    
    # Less than a specified CL limit
    CL_limit = segment.CL_limit 
    CL_con = (CL_limit  - state.conditions.aerodynamics.lift_coefficient[:,0])/CL_limit
    
    # Altitudes are greater than 0
    alt_con = state.conditions.freestream.altitude[:,0]/segment.altitude_end
    
    constraints = np.concatenate((time_con,CL_con,alt_con))
    
    return constraints

## @ingroup Methods-Missions-Segments
def get_problem_pyopt(unknowns,(segment,state)):
    """ Runs the mission and obtains the objective and all constraints. This is formatted for pyopt
    
        Assumptions:
        Time only goes forward
        CL is less than a specified limit
        All altitudes are greater than zero
        
        Inputs:
        state.unknowns      [Data]
    
        Outputs:
        obj                 [float]
        con                 [array]
        fail                [boolean]

        Properties Used:
        N/A
                                
    """       
    
    if isinstance(unknowns,array_type):
        state.unknowns.unpack_array(unknowns)
    else:
        state.unknowns = unknowns
        
    if not np.all(state.inputs_last == state.unknowns):       
        segment.process.iterate(segment,state)
        
    obj      = state.objective_value
    
    # Time goes forward, not backward
    t_final  = state.conditions.frames.inertial.time[-1,0]
    time_con = (state.conditions.frames.inertial.time[1:,0] - state.conditions.frames.inertial.time[0:-1,0])/t_final
    
    # Less than a specified CL limit
    CL_limit = segment.CL_limit 
    CL_con   = (CL_limit  - state.conditions.aerodynamics.lift_coefficient[:,0])/CL_limit
    
    # Altitudes are greater than 0
    alt_con = state.conditions.freestream.altitude[:,0]/segment.altitude_end
    
    # Put the equality and inequality constraints together
    constraints = np.concatenate((state.constraint_values,time_con,CL_con,alt_con))
    
    obj   = state.objective_value
    const = constraints.tolist()
    fail  = np.array(np.isnan(obj.tolist()) or np.isnan(np.array(const).any())).astype(int)    
    
    return obj,const,fail