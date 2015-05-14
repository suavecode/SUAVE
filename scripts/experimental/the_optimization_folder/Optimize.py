# Optimize.py
# 
# Created:  May 2015, E. Botero
# Modified: 

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import SUAVE
from SUAVE.Core import Units, Data

import Vehicles
import Analyses
import Missions
import Procedure

# TODO
Interface = Data()

# ----------------------------------------------------------------------        
#   Run the whole thing
# ----------------------------------------------------------------------  
def main():

    problem = setup()


    return

# ----------------------------------------------------------------------        
#   Inputs, Objective, & Constraints
# ----------------------------------------------------------------------  

def setup():

    problem = Interface # Change me

    # -------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------

    # [ tag , initial, [lb,ub], scaling, units ]
    problem.inputs = [
        [ 'aspect_ratio'    ,   10.    , (     5.    ,    20.   ) ,    10.  ,     Units.less], 
        [ 'reference_area'  ,   125.   , (    70.    ,   200.   ) ,   125.  , Units.meter**2],
        [ 'sweep'           ,    25.   , (     0.    ,    60.   ) ,    25.  ,  Units.degrees],
        [ 'design_thrust'   , 24000.   , ( 10000.    , 35000.   ) , 24000.  ,   Units.newton],
        [ 'wing_thickness'  ,     0.11 , (     0.07  ,     0.20 ) ,      .11,     Units.less],
        [ 'MTOW'            , 79000.   , ( 60000.    ,100000.   ) , 79000.  ,       Units.kg],
        [ 'MZFW'            , 59250.   , ( 30000.    ,100000.   ) , 59250.  ,     Units.less], 
    ]
    
    
    # -------------------------------------------------------------------
    # Objective
    # -------------------------------------------------------------------

    # throw an error if the user isn't specific about wildcards
    # [ tag, scaling, units ]
    problem.objective = [
        [ 'fuel_burn', 10000, Units.kg ]
    ]
    
    
    # -------------------------------------------------------------------
    # Constraints
    # -------------------------------------------------------------------
    
    # [ tag, sense, edge, scaling, units ]
    problem.constraints =[
        [ 'takeoff_field_length' , '<',  2180., 5000.,    Units.m],
        [ 'range_short_field'    , '>',   650.,  500.,  Units.nmi],
        [ 'range_max'            , '>',  2725., 1000.,  Units.nmi],
        [ 'max_zero_fuel_margin' , '>',     0.,    1., Units.less],
        [ 'available_fuel_margin', '>'  ,   0.,    1., Units.less],
    ]
    
    
    # -------------------------------------------------------------------
    #  Aliases
    # -------------------------------------------------------------------
    
    # [ 'alias' , ['data.path1.name','data.path2.name'] ]
    problem.aliases = [
        [ 'aspect_ratio'         ,  'configs.*.wings.main_wing.aspect_ratio'      ],
        [ 'reference_area'       ,  'configs.*.wings.main_wing.reference_area'    ],
        [ 'sweep'                ,  'configs.*.wings.main_wing.sweep'             ],
        [ 'design_thrust'        ,  'configs.*.propulsors.turbo_fan.design_thrust'],
        [ 'wing_thickness'       ,  'configs.*.wings.main_wing.thickness'         ],
        [ 'MTOW'                 , ['configs.*.mass_properties.max_takeoff'      ,
                                    'configs.*.mass_properties.takeoff'          ]],
        [ 'MZFW'                 ,  'configs.*.mass_properties.max_zero_fuel'     ],
        [ 'takeoff_field_length' ,  'results.takeoff_field_length'                ],
        [ 'range_short_field'    ,  'results.range_short_field'                   ],
        [ 'range_max'            ,  'results.range_max'                           ],
        [ 'max_zero_fuel_margin' ,  'results.max_zero_fuel_margin'                ],
        [ 'available_fuel_margin',  'results.available_fuel_margin'               ],
        [ 'fuel_burn'            ,  'results.mission_fuel.fuel'                   ],
    ]
    
    
    # -------------------------------------------------------------------
    #  Vehicles
    # -------------------------------------------------------------------
    problem.vehicles = Vehicles.setup()
    
    
    # -------------------------------------------------------------------
    #  Analyses
    # -------------------------------------------------------------------
    problem.analyses = Analyses.setup(problem.vehicles)
    
    
    # -------------------------------------------------------------------
    #  Missions
    # -------------------------------------------------------------------
    problem.missions = Missions.setup(problem.analyses)
    
    
    # -------------------------------------------------------------------
    #  Procedure
    # -------------------------------------------------------------------    
    #problem.procedure = Procedure.setup()
    
    
    return problem


if __name__ == '__main__':
    main()