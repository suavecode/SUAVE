# Optimize.py
#
# Created:  Nov 2015, Carlos / Tarik
# Modified:

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units, Data
import numpy as np
import Vehicles
import Analyses
import Missions
import Procedure
import Plot_Mission
from SUAVE.Optimization.Nexus import Nexus
import SUAVE.Optimization.Package_Setups.pyopt_setup as pyopt_setup
import SUAVE.Optimization.Package_Setups.scipy_setup as scipy_setup

# ----------------------------------------------------------------------
#   Run the whole thing
# ----------------------------------------------------------------------
def main():

    pass

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

    # [ tag , initial, [lb,ub], scaling, units ]
    problem.inputs = np.array([
        [ 'wing_area'                    ,    124.8 , (    70.    ,   200.   ) ,   124.8 , Units.meter**2],
        [ 'wing_aspect_ratio'            ,     10.18, (     5.    ,    20.   ) ,    10.18,     Units.less],
        [ 'wing_sweep'                   ,    25.   , (     0.    ,    35.   ) ,    25.  ,  Units.degrees],
        [ 'wing_thickness'               ,     0.105 , (     0.07  ,     0.20 ) ,     0.105,     Units.less],
        [ 'design_thrust'                , 52700.   , ( 10000.    , 70000.   ) , 52700.  ,        Units.N],
        [ 'MTOW'                         , 79090.   , ( 20000.    ,100000.   ) , 79090.  ,       Units.kg],
        [ 'MZFW_ratio'                   ,     0.77 , (     0.6   ,     0.99 ) ,    0.77 ,     Units.less],
        [ 'flap_takeoff_angle'           ,    10.   , (     0.    ,    20.   ) ,    10.  ,  Units.degrees],
        [ 'flap_landing_angle'           ,    40.   , (     0.    ,    50.   ) ,    40.  ,  Units.degrees],
        [ 'short_field_TOW'              , 64030.   , ( 20000.    ,100000.   ) , 64030.  ,       Units.kg],
        [ 'design_TOW'                   , 68520.   , ( 20000.    ,100000.   ) , 68520.  ,       Units.kg],
        [ 'noise_takeoff_speed_increase' ,    10.0  , (    10.    ,    20.   ) ,    10.0 ,     Units.knots],
        [ 'noise_cutback_altitude'       ,   304.8  , (   240.    ,   400.   ) ,   304.8 ,    Units.meter],
    ])

    # -------------------------------------------------------------------
    #  Objective
    # -------------------------------------------------------------------

    problem.objective = np.array([

        [ 'fuel_burn', 10000, Units.kg ],

        [ 'noise_cumulative_margin', 17, Units.less ],

    ])


    # -------------------------------------------------------------------
    # Constraints
    # -------------------------------------------------------------------

    # [ tag, sense, edge, scaling, units ]
    problem.constraints = np.array([
        [ 'MZFW consistency' , '>' , 0. , 10 , Units.less],
        [ 'design_range_fuel_margin' , '>', 0., 10, Units.less],
        [ 'short_field_fuel_margin' , '>' , 0. , 10, Units.less],
        [ 'max_range_fuel_margin' , '>' , 0. , 10, Units.less], #0.1
##        [ 'MZFW consistency' , '<' , 10. , 10 , Units.less],
##        [ 'design_range_fuel_margin' , '<', 10., 10, Units.less],
##        [ 'short_field_fuel_margin' , '<' , 10. , 10, Units.less],
##        [ 'max_range_fuel_margin' , '<' , 10. , 10, Units.less], #0.1
        [ 'wing_span' , '<', 35.9664, 35.9664, Units.less],
        [ 'noise_flyover_margin' , '>', 0. , 10., Units.less],
        [ 'noise_sideline_margin' , '>', 0. , 10. , Units.less],
        [ 'noise_approach_margin' , '>', 0., 10., Units.less],
        [ 'takeoff_field_length' , '<', 1985., 1985., Units.meters],
        [ 'landing_field_length' , '<', 1385., 1385., Units.meters],
        [ '2nd_segment_climb_max_range' , '>', 0.024, 0.024, Units.less],
        [ '2nd_segment_climb_short_field' , '>', 0.024, 0.024, Units.less],
        [ 'max_throttle' , '<', 1., 1., Units.less],
        [ 'short_takeoff_field_length' , '<', 1330., 1330., Units.meters],
        [ 'noise_cumulative_margin' , '>', 10., 10., Units.less],
    ])

    # -------------------------------------------------------------------
    #  Aliases
    # -------------------------------------------------------------------

    # [ 'alias' , ['data.path1.name','data.path2.name'] ]

    problem.aliases = [
        [ 'wing_area'                        ,   ['vehicle_configurations.*.wings.main_wing.areas.reference',
                                                  'vehicle_configurations.*.reference_area'                            ]],
        [ 'wing_aspect_ratio'                ,    'vehicle_configurations.*.wings.main_wing.aspect_ratio'               ],
        [ 'wing_incidence'                   ,    'vehicle_configurations.*.wings.main_wing.twists.root'                ],
        [ 'wing_tip_twist'                   ,    'vehicle_configurations.*.wings.main_wing.twists.tip'                 ],
        [ 'wing_sweep'                       ,    'vehicle_configurations.*.wings.main_wing.sweep'                      ],
        [ 'wing_thickness'                   ,    'vehicle_configurations.*.wings.main_wing.thickness_to_chord'         ],
        [ 'wing_taper'                       ,    'vehicle_configurations.*.wings.main_wing.taper'                      ],
        [ 'wing_location'                    ,    'vehicle_configurations.*.wings.main_wing.origin[0]'                  ],
        [ 'horizontal_tail_area'             ,    'vehicle_configurations.*.wings.horizontal_stabilizer.areas.reference'],
        [ 'horizontal_tail_aspect_ratio'     ,    'vehicle_configurations.*.wings.horizontal_stabilizer.aspect_ratio'   ],
        [ 'vertical_tail_area'               ,    'vehicle_configurations.*.wings.vertical_stabilizer.areas.reference'  ],
        [ 'vertical_tail_aspect_ratio'       ,    'vehicle_configurations.*.wings.vertical_stabilizer.aspect_ratio'     ],
        [ 'design_thrust'                    ,    'vehicle_configurations.*.propulsors.turbo_fan.thrust.total_design'   ],
        [ 'MTOW'                             ,   ['vehicle_configurations.*.mass_properties.takeoff'   ,
                                                  'vehicle_configurations.*.mass_properties.max_takeoff'               ]],
        [ 'design_TOW'                       ,    'vehicle_configurations.base.mass_properties.takeoff'                 ],
        [ 'short_field_TOW'                  ,    'vehicle_configurations.short_field_takeoff.mass_properties.takeoff'  ],
        #[ 'MZFW'                             ,    'vehicle_configurations.*.mass_properties.max_zero_fuel'              ],
        [ 'flap_takeoff_angle'               ,    ['vehicle_configurations.takeoff.wings.main_wing.flaps.angle',
                                                   'vehicle_configurations.short_field_takeoff.wings.main_wing.flaps.angle']],
        #[ 'flap_short_field_angle'           ,    'vehicle_configurations.short_field_takeoff.wings.main_wing.flaps.angle'],
        [ 'flap_landing_angle'               ,    'vehicle_configurations.landing.wings.main_wing.flaps.angle'          ],
        [ 'slat_takeoff_angle'               ,    ['vehicle_configurations.takeoff.wings.main_wing.slats.angle',
                                               'vehicle_configurations.short_field_takeoff.wings.main_wing.slats.angle']],
        [ 'slat_landing_angle'               ,    'vehicle_configurations.landing.wings.main_wing.slats.angle'          ],
        [ 'fuel_burn'                        ,    'summary.base_mission_fuelburn'                                                   ],
        [ 'wing_span'                        ,    'vehicle_configurations.base.wings.main_wing.spans.projected'         ],
        #[ 'engine_fan_diameter'              ,    'vehicle_configurations.base.turbofan.nacelle_diameter'               ],
        [ 'noise_approach_margin'            ,    'summary.noise_approach_margin'                                       ],
        [ 'noise_sideline_margin'            ,    'summary.noise_sideline_margin'                                       ],
        [ 'noise_flyover_margin'             ,    'summary.noise_flyover_margin'                                        ],
        [ 'static_stability'                 ,    'summary.static_stability'                                            ],
        [ 'vertical_tail_volume_coefficient' ,    'summary.vertical_tail_volume_coefficient'                            ],
        [ 'horizontal_tail_volume_coefficient',   'summary.horizontal_tail_volume_coefficient'                          ],
        [ 'wing_max_cl_norm'                 ,    'summary.maximum_cl_norm'                                             ],
        [ 'design_range_fuel_margin'         ,    'summary.design_range_fuel_margin'                                    ],
        [ 'takeoff_field_length'             ,    'summary.takeoff_field_length'                                        ],
        [ 'landing_field_length'             ,    'summary.landing_field_length'                                        ],
        [ 'short_takeoff_field_length'       ,    'summary.short_takeoff_field_length'                                  ],
        [ '2nd_segment_climb_max_range'      ,    'summary.second_segment_climb_gradient_takeoff'                       ],
        [ '2nd_segment_climb_short_field'    ,    'summary.second_segment_climb_gradient_short_field'                   ],
        [ 'max_throttle'                     ,    'summary.max_throttle'                                                ],
        [ 'short_field_fuel_margin'          ,    'summary.short_field_fuel_margin'                                     ],
        [ 'max_range_fuel_margin'            ,    'summary.max_range_fuel_margin'                                       ],
        [ 'max_range'                        ,    'missions.max_range_distance'                                         ],
        [ 'MZFW consistency'                 ,    'summary.MZFW_consistency'                                            ],
        [ 'MZFW_ratio'                       ,    'MZFW_ratio'                                                          ],
        [ 'noise_takeoff_speed_increase'     ,    'noise_V2_increase'                                                   ],
        [ 'noise_cutback_altitude'           ,    'missions.takeoff.segments.climb.altitude_end'                        ],
        [ 'noise_cumulative_margin'          ,    'summary.noise_margin'                                                ],
        [ 'weighted_sum_objective'           ,    'summary.weighted_sum_objective'                                      ],
    ]

    # -------------------------------------------------------------------
    #  Vehicles
    # -------------------------------------------------------------------
    nexus.vehicle_configurations = Vehicles.setup()


    # -------------------------------------------------------------------
    #  Analyses
    # -------------------------------------------------------------------
    nexus.analyses = Analyses.setup(nexus.vehicle_configurations)


    # -------------------------------------------------------------------
    #  Missions
    # -------------------------------------------------------------------
    nexus.missions = Missions.setup(nexus.analyses)


    # -------------------------------------------------------------------
    #  Procedure
    # -------------------------------------------------------------------
    nexus.procedure = Procedure.setup()

    # -------------------------------------------------------------------
    #  Summary
    # -------------------------------------------------------------------
    nexus.summary = Data()

    return nexus


if __name__ == '__main__':


    problem = setup()

    n_des_var = 13

    var = np.zeros(n_des_var)

##    var = [128.4,10.0746256149533,35,0.114,51860,70000,0.75,10.2,46,70000,70000,11.5,299]
##    var = [134,9.60532850235224,31,0.155,57220,70000,0.75,19.6,36.6,70000,70000,10.5,400]
##    var = [133.6,9.68249946826347,35,0.131,57220,70000,0.75,19.8,18,70000,70000,10,377]
##    var = [134.4,9.62486554285714,35,0.155,56020,70000,0.75,19.8,21.6,70000,70000,11.5,393]
    #var = [131.6,	9.8296499161,	35.0,	0.118,	50400.0,	70000.0,	0.75,	12.0,	41.8,	70000.0,	70000.0,	10.0,	359.0]
    #var = [133.8,	9.6680263749,	35.0,	0.113,	51160.0,	70000.0,	0.75,	8.8,	50.0,	70000.0,	70000.0,	10.5,	347.0]
    #var = [132.6,	9.7555198262,	35.0,	0.124,	50180.0,	70000.0,	0.75,	8.2,	50.0,	70000.0,	70000.0,	11.5,	400.0]
    #var = [128.4,	10.074625615,	35.0,	0.114,	51860.0,	70000.0,	0.75,	10.2,	46.0,	70000.0,	70000.0,	11.5,	299.0]
    var = [134.6,	9.6105641082,	35.0,	0.123,	49200.0,	70000.0,	0.75,	6.6,	30.0,	70000.0,	70000.0,	11.5,	283.0]
    #var = [132.8,	9.7408277783,	34.8,	0.135,	49140.0,	70000.0,	0.75,	11.4,	17.2,	70000.0,	70000.0,	10.0,	400.0]
    #var =  [133.8,	9.6680263749,	34.8,	0.133,	49140.0,	70000.0,	0.75,	18.8,	32.2,	70000.0,	70000.0,	10.5,	399.0]
    #var = [136.0,	9.5116318306,	35.0,	0.141,	49960.0,	70000.0,	0.75,	19.8,	22.6,	70000.0,	70000.0,	11.5,	393.0]

##    AR = 35.9664*35.9664 / var[0] * var[1]

##    var[ 0] = 		SW			#                 'wing_area'
##    var[ 1] =  		AR           #                 'wing_aspect_ratio'
##    var[ 2] =  		SWEEP          #                 'wing_sweep'
##    var[ 3] =  	 	wing_tc           #                 'wing_thickness'
##    var[ 4] =	    THRUST              #                 'design_thrust'
##    var[ 5] =	    MTOW              #                 'MTOW'
##    var[ 6] =	 	MZFW_ratio            #                 'MZFW_ratio'
##    var[ 7] =	    FLAP_TO              #                 'flap_takeoff_angle'
##    var[ 8] =	    FLAP_LND              #                 'flap_landing_angle'
##    var[ 9] =	    SF_TOW              #                 'short_field_TOW'
##    var[10] =	    DESIGN_TOW              #                 'design_TOW'
##    var[11] =	    NOISE_dV2              #                 'noise_takeoff_speed_increase'
##    var[12] =	    HP_cutback            #                 'noise_cutback_altitude'

##    var[ 0] = 		134.47 #SW			#                 'wing_area'
##    var[ 1] =  		35.9664*35.9664 / var[ 0]  #AR           #                 'wing_aspect_ratio'
##    var[ 2] =  		35. #SWEEP          #                 'wing_sweep'
##    var[ 3] =  	 	0.155 #wing_tc           #                 'wing_thickness'
##    var[ 4] =	    56020. #THRUST              #                 'design_thrust'
##    var[ 5] =	    70000. #MTOW              #                 'MTOW'
##    var[ 6] =	 	0.75 #MZFW_ratio            #                 'MZFW_ratio'
##    var[ 7] =	    19.8 #FLAP_TO              #                 'flap_takeoff_angle'
##    var[ 8] =	    21.6 #FLAP_LND              #                 'flap_landing_angle'
##    var[ 9] =	    70000. #SF_TOW              #                 'short_field_TOW'
##    var[10] =	    70000. #DESIGN_TOW              #                 'design_TOW'
##    var[11] =	    11.5 #NOISE_dV2              #                 'noise_takeoff_speed_increase'
##    var[12] =	    393. #HP_cutback            #                 'noise_cutback_altitude'

    input_vec = var / problem.optimization_problem.inputs[:,3]

    problem.objective(input_vec)

    constraints = problem.all_constraints() * problem.optimization_problem.constraints[:,3]
    objectives  = problem.objective()       * problem.optimization_problem.objective[:,1]

    fuel_burn               = objectives[0]
    noise_cumulative_margin = objectives[1]

##    MZFW_consistency                =	constraints[0]
##    design_range_fuel_margin        =	constraints[1]
##    short_field_fuel_margin         =	constraints[2]
##    max_range_fuel_margin           =	constraints[3]
##    wing_span                       =	constraints[4]
##    noise_flyover_margin            =	constraints[5]
##    noise_sideline_margin           =	constraints[6]
##    noise_approach_margin           =	constraints[7]
##    takeoff_field_length            =	constraints[8]
##    landing_field_length            =	constraints[9]
##    sec_segment_climb_max_range	    =	constraints[10]
##    sec_segment_climb_short_field	=	constraints[11]
##    max_throttle	                =	constraints[12]
##    short_takeoff_field_length	    =	constraints[13]
##    noise_cumulative_margin	        =	constraints[14]

    print constraints
    print objectives
    
    print "Fuel Burn = ", fuel_burn
    print "Noise Margin = ", noise_cumulative_margin
    