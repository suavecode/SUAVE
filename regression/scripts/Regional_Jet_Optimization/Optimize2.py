# Optimize.py
# Created:  Feb 2016, M. Vegh
# Modified: Nov 2016, T. MacDonald

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import SUAVE
from SUAVE.Core import Units, Data
import numpy as np
import Analyses2
import Missions2
import Procedure2
import Plot_Mission2
import matplotlib.pyplot as plt
from SUAVE.Optimization import Nexus, carpet_plot
import SUAVE.Optimization.Package_Setups.scipy_setup as scipy_setup
import sys
sys.path.append('../Vehicles')
from Embraer_190 import vehicle_setup, configs_setup
# ----------------------------------------------------------------------        
#   Run the whole thing
# ----------------------------------------------------------------------  
def main():
    problem = setup()
    
    # Enforce the bounds
    problem.hard_bounded_inputs = True
    
    obj = problem.objective([1.,1.])
    con = problem.all_constraints([1.,1.])
    obj2 = problem.objective([0.9,1.1])
    con3 = problem.all_constraints([1.1,0.9])   
    
    print('Fuel Burn   =', obj)
    print('Fuel Margin =', con)
    print(obj2)
    print(con3)
    
    actual = Data()
    actual.obj  = 0.7104222
    actual.con  = 2.73413428
    actual.obj2 = 0.73352975
    actual.con3 = 2.88527578
    

    error = Data()
    error.obj  = (actual.obj - obj)/actual.obj
    error.con  = (actual.con - con)/actual.con
    error.obj2 = (actual.obj2 - obj2)/actual.obj2
    error.con3 = (actual.con3 - con3)/actual.con3 

    print('Fuel Burn Error   =',error.obj)
    print('Fuel Margin Error =',error.con)
    
    for k,v in list(error.items()):
        assert(np.abs(v)<1e-6)     
        
    return

# ----------------------------------------------------------------------        
#   Inputs, Objective, & Constraints
# ----------------------------------------------------------------------  

def setup():

    nexus = Nexus()
    problem = Data()
    nexus.optimization_problem = problem

    # -------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------

    #   [ tag                            , initial, (lb,ub)             , scaling , units ]
    problem.inputs = np.array([
        [ 'wing_area'                    ,  95    ,    90. ,   130.    ,   100. , 1*Units.meter**2],
        [ 'cruise_altitude'              ,  11    ,    9.  ,    14.    ,   10.  , 1*Units.km],
    ],dtype=object)
    
    # -------------------------------------------------------------------
    # Objective
    # -------------------------------------------------------------------

    # throw an error if the user isn't specific about wildcards
    # [ tag, scaling, units ]
    problem.objective = np.array([
        [ 'fuel_burn', 10000, 1*Units.kg ]
    ],dtype=object)
    
    # -------------------------------------------------------------------
    # Constraints
    # -------------------------------------------------------------------
    
    # [ tag, sense, edge, scaling, units ]
    problem.constraints = np.array([
        [ 'design_range_fuel_margin' , '>', 0., 1E-1, 1*Units.less], #fuel margin defined here as fuel 
    ],dtype=object)
    
    # -------------------------------------------------------------------
    #  Aliases
    # -------------------------------------------------------------------
    
    # [ 'alias' , ['data.path1.name','data.path2.name'] ]

    problem.aliases = [
        [ 'wing_area'                        ,   ['vehicle_configurations.*.wings.main_wing.areas.reference',
                                                  'vehicle_configurations.*.reference_area'                    ]],
        [ 'cruise_altitude'                  , 'missions.base.segments.climb_5.altitude_end'                    ],
        [ 'fuel_burn'                        ,    'summary.base_mission_fuelburn'                               ],
        [ 'design_range_fuel_margin'         ,    'summary.max_zero_fuel_margin'                                ],
    ]    
    
    # -------------------------------------------------------------------
    #  Vehicles
    # -------------------------------------------------------------------
    vehicle = vehicle_setup()
    nexus.vehicle_configurations = configs_setup(vehicle)
    
    # -------------------------------------------------------------------
    #  Analyses
    # -------------------------------------------------------------------
    nexus.analyses = Analyses2.setup(nexus.vehicle_configurations)
    
    
    # -------------------------------------------------------------------
    #  Missions
    # -------------------------------------------------------------------
    nexus.missions = Missions2.setup(nexus.analyses)
    
    
    # -------------------------------------------------------------------
    #  Procedure
    # -------------------------------------------------------------------    
    nexus.procedure = Procedure2.setup()
    
    # -------------------------------------------------------------------
    #  Summary
    # -------------------------------------------------------------------    
    nexus.summary = Data()    
    
    return nexus
    
def variable_sweep(problem):    
    number_of_points=5
    outputs=carpet_plot(problem, number_of_points, 0, 0)  #run carpet plot, suppressing default plots
    inputs =outputs.inputs
    objective=outputs.objective
    constraints=outputs.constraint_val
    plt.figure(0)
    CS = plt.contourf(inputs[0,:],inputs[1,:], objective, 20, linewidths=2)
    cbar = plt.colorbar(CS)
    
    cbar.ax.set_ylabel('fuel burn (kg)')
    CS_const=plt.contour(inputs[0,:],inputs[1,:], constraints[0,:,:])
    plt.clabel(CS_const, inline=1, fontsize=10)
    cbar = plt.colorbar(CS_const)
    cbar.ax.set_ylabel('fuel margin')
    
    
    
    plt.xlabel('wing area (m^2)')
    plt.ylabel('cruise_altitude (km)')
    
    '''
    #now plot optimization path (note that these data points were post-processed into a plottable format)
    wing_1  = [95          ,	95.00000149 ,	95          ,	95          ,	95.00000149 ,	95          ,	95          ,	95.00000149 ,	95          ,	106.674165  ,	106.6741665 ,	106.674165  ,	106.674165  ,	106.6741665 ,	106.674165  ,	106.674165  ,	106.6741665 ,	106.674165  ,	105.6274294 ,	105.6274309 ,	105.6274294 ,	105.6274294 ,	105.6274309 ,	105.6274294 ,	105.6274294 ,	105.6274309 ,	105.6274294 ,	106.9084316 ,	106.9084331 ,	106.9084316 ,	106.9084316 ,	106.9084331 ,	106.9084316 ,	106.9084316 ,	106.9084331 ,	106.9084316 ,	110.520489  ,	110.5204905 ,	110.520489  ,	110.520489  ,	110.5204905 ,	110.520489  ,	110.520489  ,	110.5204905 ,	110.520489  ,	113.2166831 ,	113.2166845 ,	113.2166831 ,	113.2166831 ,	113.2166845 ,	113.2166831 ,	113.2166831 ,	113.2166845 ,	113.2166831 ,	114.1649262 ,	114.1649277 ,	114.1649262 ,	114.1649262 ,	114.1649277 ,	114.1649262 ,	114.1649262 ,	114.1649277 ,	114.1649262 ,	114.2149828]
    alt_1   = [11.0              ,	11.0              ,	11.000000149011612,	11.0              ,	11.0              ,	11.000000149011612,	11.0              ,	11.0              ,	11.000000149011612,	9.540665954351425 ,	9.540665954351425 ,	9.540666103363037 ,	9.540665954351425 ,	9.540665954351425 ,	9.540666103363037 ,	9.540665954351425 ,	9.540665954351425 ,	9.540666103363037 ,	10.023015652305284,	10.023015652305284,	10.023015801316896,	10.023015652305284,	10.023015652305284,	10.023015801316896,	10.023015652305284,	10.023015652305284,	10.023015801316896,	10.190994033521863,	10.190994033521863,	10.190994182533474,	10.190994033521863,	10.190994033521863,	10.190994182533474,	10.190994033521863,	10.190994033521863,	10.190994182533474,	10.440582829327589,	10.440582829327589,	10.4405829783392  ,	10.440582829327589,	10.440582829327589,	10.4405829783392  ,	10.440582829327589,	10.440582829327589,	10.4405829783392  ,	10.536514606250261,	10.536514606250261,	10.536514755261873,	10.536514606250261,	10.536514606250261,	10.536514755261873,	10.536514606250261,	10.536514606250261,	10.536514755261873,	10.535957839878783,	10.535957839878783,	10.535957988890395,	10.535957839878783,	10.535957839878783,	10.535957988890395,	10.535957839878783,	10.535957839878783,	10.535957988890395,	10.52829047]
    wing_2  = [128        ,	128.0000015,	128        ,	128        ,	128.0000015,	128        ,	128        ,	128.0000015,	128        ,	130        ,	130.0000015,	130        ,	130        ,	130.0000015,	130        ,	130        ,	130.0000015,	130        ,	122.9564124,	122.9564139,	122.9564124,	122.9564124,	122.9564139,	122.9564124,	122.9564124,	122.9564139,	122.9564124,	116.5744347,	116.5744362,	116.5744347,	116.5744347,	116.5744362,	116.5744347,	116.5744347,	116.5744362,	116.5744347,	116.3530891,	116.3530906,	116.3530891,	116.3530891,	116.3530906,	116.3530891,	116.3530891,	116.3530906,	116.3530891]
    alt_2   = [13.8,	13.799999999999999,	13.80000014901161,	13.799999999999999,	13.799999999999999,	13.80000014901161,	13.799999999999999,	13.799999999999999,	13.80000014901161,	11.302562430674953,	11.302562430674953,	11.302562579686565,	11.302562430674953,	11.302562430674953,	11.302562579686565,	11.302562430674953,	11.302562430674953,	11.302562579686565,	11.158808932491421,	11.158808932491421,	11.158809081503033,	11.158808932491421,	11.158808932491421,	11.158809081503033,	11.158808932491421,	11.158808932491421,	11.158809081503033,	11.412913394878741,	11.412913394878741,	11.412913543890353,	11.412913394878741,	11.412913394878741,	11.412913543890353,	11.412913394878741,	11.412913394878741,	11.412913543890353,	11.402627869388722,	11.402627869388722,	11.402628018400334,	11.402627869388722,	11.402627869388722,	11.402628018400334,	11.402627869388722,	11.402627869388722,	11.402628018400334]

    
    opt_1   = plt.plot(wing_1, alt_1, label='optimization path 1')
    init_1  = plt.plot(wing_1[0], alt_1[0], 'ko')
    final_1 = plt.plot(wing_1[-1], alt_1[-1], 'kx')
    
    opt_2   = plt.plot(wing_2, alt_2, 'k--', label='optimization path 2')
    init_2  = plt.plot(wing_2[0], alt_2[0], 'ko', label= 'initial points')
    final_2 = plt.plot(wing_2[-1], alt_2[-1], 'kx', label= 'final points')
    '''
    plt.legend(loc='upper left')  
    plt.show()    
    
      

    return
    

if __name__ == '__main__':
    main()
    
    
