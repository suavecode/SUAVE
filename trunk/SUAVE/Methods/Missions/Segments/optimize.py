# optimize.py
# 
# Created:  Dec 2016, E. Botero
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import scipy.optimize as opt
import numpy as np

from SUAVE.Core.Arrays import array_type
from SUAVE.Core import Units
import copy

import pyOpt
import pyOpt.pySNOPT

# ----------------------------------------------------------------------
#  Converge Root
# ----------------------------------------------------------------------

def converge_opt(segment,state):
    
    # pack up the array
    unknowns = state.unknowns.pack_array()
    
    #segment_points = state.numerics.number_control_points
    
    # Have the optimizer call the wrapper
    obj   = lambda unknowns:get_objective(unknowns,(segment,state))   
    econ  = lambda unknowns:get_econstraints(unknowns,(segment,state)) 
    iecon = lambda unknowns:get_ieconstraints(unknowns,(segment,state)) 
    
    bnds = make_bnds(unknowns, state)
    
    #unknowns = opt.fmin_slsqp(obj,unknowns,f_eqcons=econ, f_ieqcons=iecon, bounds=bnds,iter=2000)
    
    unknowns = opt.fmin_slsqp(obj,unknowns,f_eqcons=econ,bounds=bnds,iter=2000)
    
    #unknowns = opt.fmin_slsqp(obj,unknowns,f_eqcons=econ,f_ieqcons=iecon,bounds=bnds,iter=2000)
    
    #opt_prob = pyOpt.Optimization('SUAVE',obj)
    #opt_prob.addObj('Something')

    #for ii in xrange(0,len(unknowns)):
        #lbd = (bnds[ii][0])
        #ubd = (bnds[ii][1])
        #vartype = 'c'
        #opt_prob.addVar(str(ii),vartype,lower=lbd,upper=ubd,value=unknowns[ii])  
        

    ## Setup constraints  
    #for ii in xrange(0,2*segment_points):
        #opt_prob.addCon(str(ii), type='e', equal=0.)     
        
    #print opt_prob

    #opt = pyOpt.pySNOPT.SNOPT()    
    
    #outputs = opt(opt_prob) 
    
    #print outputs
    
    #print opt_prob.solution(0)
                      
    return
    
# ----------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------
    

def get_objective(unknowns,(segment,state)):
    if isinstance(unknowns,array_type):
        state.unknowns.unpack_array(unknowns)
    else:
        state.unknowns = unknowns
        
    if not np.all(state.inputs_last == state.unknowns):       
        segment.process.iterate(segment,state)
        
    objective = state.objective_value
    
    #print unknowns
    
    #print objective
    
    return objective

def get_econstraints(unknowns,(segment,state)):
    
    if isinstance(unknowns,array_type):
        state.unknowns.unpack_array(unknowns)
    else:
        state.unknowns = unknowns
        
    if not np.all(state.inputs_last == state.unknowns):       
        segment.process.iterate(segment,state)

    constraints = state.constraint_values
    
    constraints2 = np.array([(segment.air_speed_start -state.conditions.freestream.velocity[0,0])/segment.air_speed_start])
    
    constraints = np.concatenate((constraints,constraints2))
    print constraints
        
    return constraints


def make_bnds(unknowns,state):
    
    ones    = state.ones_row(1)
    ones_m1 = state.ones_row_m1(1)
    ones_m2 = state.ones_row_m2(1)
    
    throttle_bnds = ones*(0.,1.)
    body_angle    = ones*(0. * Units.degrees, 60. * Units.degrees)
    gamma         = ones_m1*(0. * Units.degrees, 60. * Units.degrees)
    vels          = ones_m1*(0.,1.e20)
    
    bnds = np.vstack([vels,throttle_bnds,gamma,body_angle])
    
    bnds = list(map(tuple, bnds))
    
    return bnds


def get_ieconstraints(unknowns,(segment,state)):
    
    if isinstance(unknowns,array_type):
        state.unknowns.unpack_array(unknowns)
    else:
        state.unknowns = unknowns
        
    if not np.all(state.inputs_last == state.unknowns):       
        segment.process.iterate(segment,state)
    
    
    #a  = state.conditions.frames.inertial.acceleration_vector[:,2] 
    
    #constraints2 = np.concatenate([1.5*9.81 + a,9.81-a])
    
    #constraints = state.conditions.frames.inertial.time[1:,0] - state.conditions.frames.inertial.time[0:-1,0] 
    
    
    #constraints = np.concatenate((constraints))
    
    return constraints

