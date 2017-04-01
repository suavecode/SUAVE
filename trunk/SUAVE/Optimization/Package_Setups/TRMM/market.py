"""
Manages global variables used during trust region optimization

Rick Fenrich 2/22/17
"""

import multiprocessing
import copy
import os

def init():

    global K, T, TRC, S
    global HIST, SURR_HIST, TR_HIST
    global EVAL, SURR_EVAL
    global USER_DATA
    global ROOT_DIR
    
    K = 0 # iteration index
    T = 0 # trust region center index
    TRC = 0 # trust region center
    S = 0 # shared data index
    
    HIST = {} # history for truth function evaluations
    SURR_HIST = {} # history for evaluation of surrogate models (all fidelity levels)
    TR_HIST = {} # trust region history
    
    EVAL = {} # number of truth function evals
    EVAL_DUP = {} # number of (detected) duplicate truth function evals
    SURR_EVAL = {} # number of model function evals
    
    USER_DATA = {} # user can add shared data to this dictionary
    ROOT_DIR = os.getcwd()
        
def add_user_data(key,value):
    global USER_DATA
    if key in USER_DATA: # if data already exists, make a copy of it
        USER_DATA['_'+key] = copy.copy(USER_DATA[key])
        USER_DATA[key] = copy.copy(value)
    else:
        USER_DATA[key] = copy.copy(value)
    
def assign_to_history(tag,x,y,val):
    '''Record response for truth function evaluation 
    at x and y for truth function specified by tag'''
    global HIST, EVAL
    HIST[tag]['x'].append(copy.copy(x))
    HIST[tag]['y'].append(copy.copy(y))
    HIST[tag]['response'].append(copy.copy(val))
    EVAL[tag] += 1

def assign_to_surrogate_history(l,k,x,y,obj,con):
    '''Record response of objective and constraints
    for evaluation of surrogate function of level l
    at x and y, increment function evaluation'''
    global SURR_HIST, SURR_EVAL
    SURR_HIST[l]['iter'].append(copy.copy(k))
    SURR_HIST[l]['x'].append(copy.copy(x))
    SURR_HIST[l]['y'].append(copy.copy(y))
    SURR_HIST[l]['objective'].append(copy.copy(obj))
    SURR_HIST[l]['constraints'].append(copy.copy(con))
    SURR_EVAL[l] += 1
        
def assign_to_trust_region_history(k,t,center,size):
    '''Record trust region information at iteration
    k including center index t, center and size'''
    global TR_HIST
    TR_HIST['iter'].append(copy.copy(k))
    TR_HIST['center_index'].append(copy.copy(t))
    TR_HIST['center'].append(copy.copy(center))
    TR_HIST['trSize'].append(copy.copy(size))
    
def get_shared_data_index():
    global S
    return copy.copy(S)
    
def get_trust_region_center():
    global TRC
    return copy.copy(TRC)
  
def get_trust_region_center_index():
    global T
    return copy.copy(T)
        
def get_user_data(key):
    global USER_DATA
    if key in USER_DATA:
        return copy.copy(USER_DATA[key])
    else:
        raise KeyError('key %s not found' % key)
        
def increment_iteration():
    global K
    K += 1
    return copy.copy(K)
    
def increment_shared_data_index():
    global S
    S += 1
    return copy.copy(S)
    
def increment_trust_region_center_index():
    global T
    T += 1
    return copy.copy(T)
 
def revert_user_data_update():
    global USER_DATA
    for key in USER_DATA: # update all keys from keys prefixed with '_'
        if( '_' + key in USER_DATA ):
            USER_DATA[key] = USER_DATA['_'+key]
    
def set_trust_region_center(nparray):
    global TRC
    TRC = nparray
    return copy.copy(TRC)
    
def setup_history(tagList):
    global HIST, EVAL
    for tag in tagList:
        HIST[tag] = {'x': [], 'y': [], 'response': []}
        EVAL[tag] = 0

def setup_surrogate_history(levelList):
    global SURR_HIST, SURR_EVAL
    for level in levelList:
        SURR_HIST[level] = {'iter': [], 'x': [], 'y': [], 'objective': [], 'constraints': []}
        SURR_EVAL[level] = 0
    
def setup_trust_region_history():
    global TR_HIST 
    TR_HIST = {'iter': [], 'center_index': [], 'center': [], 'trSize': []}
