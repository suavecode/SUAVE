# Optimize.py
#
# Created:  Nov 2015, Carlos / Tarik
# Modified: Nov 2016, T. MacDonald

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units, Data
import numpy as np
import Noise_Analyses
import Noise_Missions
import Noise_Procedure 
from SUAVE.Optimization.Nexus import Nexus 
import sys
sys.path.append('../Vehicles')
# the analysis functions

from Boeing_737 import vehicle_setup, configs_setup


# ----------------------------------------------------------------------
#   Run the whole thing
# ----------------------------------------------------------------------
def main():
    # New Regression Flag
    generate_new_truth_data = False # To be left false unless changing noise model

    # Problem Setup
    problem   = setup(generate_new_truth_data)
    var       = np.array([134.6,9.6105641082,35.0,0.123,49200.0,70000.0,0.75,6.6,30.0,70000.0,70000.0,11.5,283.0])
    input_vec = var / problem.optimization_problem.inputs[:,3]

    # Problem Objective
    problem.objective(input_vec)
    objectives  = problem.objective()* problem.optimization_problem.objective[:,1]

    # Compare with truth values
    noise_cumulative_margin        = objectives[0]
    actual                         = Data()
    actual.noise_cumulative_margin = 21.071910892291967


    error                         = Data()
    error.noise_cumulative_margin = abs(actual.noise_cumulative_margin - noise_cumulative_margin)/actual.noise_cumulative_margin
    
    print('noise_cumulative_margin_error=', noise_cumulative_margin)
    
    print(error.noise_cumulative_margin)
    print(error)
    for k,v in list(error.items()):
        assert(np.abs(v)<1e-6) 
        
    return
        

# ----------------------------------------------------------------------
#   Inputs, Objective, & Constraints
# ----------------------------------------------------------------------

def setup(generate_new_truth_data):

    nexus = Nexus()
    problem = Data()
    nexus.optimization_problem = problem

    # -------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------

    # [ tag , initial, [lb,ub], scaling, units ]
    problem.inputs = np.array([
        [ 'wing_area'                    ,    124.8 ,     70.    ,   200.    ,   124.8 , 1*Units.meter**2],
        [ 'wing_aspect_ratio'            ,     10.18,      5.    ,    20.    ,    10.18,     1*Units.less],
        [ 'wing_sweep'                   ,    25.   ,      0.    ,    35.    ,    25.  ,  1*Units.degrees],
        [ 'wing_thickness'               ,    0.105 ,     0.07   ,     0.20  ,    0.105,    1*Units.less ],
        [ 'design_thrust'                , 52700.   ,  10000.    , 70000.    , 52700.  ,        1*Units.N],
        [ 'MTOW'                         , 79090.   ,  20000.    ,100000.    , 79090.  ,       1*Units.kg],
        [ 'MZFW_ratio'                   ,     0.77 ,      0.6   ,     0.99  ,    0.77 ,     1*Units.less],
        [ 'flap_takeoff_angle'           ,    10.   ,      0.    ,    20.    ,    10.  ,  1*Units.degrees],
        [ 'flap_landing_angle'           ,    40.   ,      0.    ,    50.    ,    40.  ,  1*Units.degrees],
        [ 'short_field_TOW'              , 64030.   ,  20000.    ,100000.    , 64030.  ,       1*Units.kg],
        [ 'design_TOW'                   , 68520.   ,  20000.    ,100000.    , 68520.  ,       1*Units.kg],
        [ 'noise_takeoff_speed_increase' ,    10.0  ,     10.    ,    20.    ,    10.0 ,    1*Units.knots],
        [ 'noise_cutback_altitude'       ,   304.8  ,    240.    ,   400.    ,   304.8 ,    1*Units.meter],
    ],dtype=object)

    # -------------------------------------------------------------------
    #  Objective
    # -------------------------------------------------------------------

    problem.objective = np.array([

        [ 'noise_cumulative_margin', 17, 1*Units.less ],

    ],dtype=object)


    # -------------------------------------------------------------------
    # Constraints
    # -------------------------------------------------------------------

    # [ tag, sense, edge, scaling, units ]
    problem.constraints = np.array([
        [ 'MZFW consistency' , '>' , 0. , 10 , 1*Units.less],
        [ 'design_range_fuel_margin' , '>', 0., 10, 1*Units.less],
        [ 'short_field_fuel_margin' , '>' , 0. , 10, 1*Units.less],
        [ 'max_range_fuel_margin' , '>' , 0. , 10, 1*Units.less], 
        [ 'wing_span' , '<', 35.9664, 35.9664, 1*Units.less],
        [ 'noise_flyover_margin' , '>', 0. , 10., 1*Units.less],
        [ 'noise_sideline_margin' , '>', 0. , 10. , 1*Units.less],
        [ 'noise_approach_margin' , '>', 0., 10., 1*Units.less],
        [ 'takeoff_field_length' , '<', 1985., 1985., 1*Units.meters],
        [ 'landing_field_length' , '<', 1385., 1385., 1*Units.meters],
        [ '2nd_segment_climb_max_range' , '>', 0.024, 0.024, 1*Units.less],
        [ '2nd_segment_climb_short_field' , '>', 0.024, 0.024, 1*Units.less],
        [ 'max_throttle' , '<', 1., 1., 1*Units.less],
        [ 'short_takeoff_field_length' , '<', 1330., 1330., 1*Units.meters],
        [ 'noise_cumulative_margin' , '>', 10., 10., 1*Units.less],
    ],dtype=object)

    # -------------------------------------------------------------------
    #  Aliases
    # -------------------------------------------------------------------


    problem.aliases = [
        [ 'wing_area'                        ,   ['vehicle_configurations.*.wings.main_wing.areas.reference',
                                                  'vehicle_configurations.*.reference_area'                            ]],
        [ 'wing_aspect_ratio'                ,    'vehicle_configurations.*.wings.main_wing.aspect_ratio'               ],
        [ 'wing_incidence'                   ,    'vehicle_configurations.*.wings.main_wing.twists.root'                ],
        [ 'wing_tip_twist'                   ,    'vehicle_configurations.*.wings.main_wing.twists.tip'                 ],
        [ 'wing_sweep'                       ,    'vehicle_configurations.*.wings.main_wing.sweeps.quarter_chord'        ],
        [ 'wing_thickness'                   ,    'vehicle_configurations.*.wings.main_wing.thickness_to_chord'         ],
        [ 'wing_taper'                       ,    'vehicle_configurations.*.wings.main_wing.taper'                      ],
        [ 'wing_location'                    ,    'vehicle_configurations.*.wings.main_wing.origin[0]'                  ],
        [ 'horizontal_tail_area'             ,    'vehicle_configurations.*.wings.horizontal_stabilizer.areas.reference'],
        [ 'horizontal_tail_aspect_ratio'     ,    'vehicle_configurations.*.wings.horizontal_stabilizer.aspect_ratio'   ],
        [ 'vertical_tail_area'               ,    'vehicle_configurations.*.wings.vertical_stabilizer.areas.reference'  ],
        [ 'vertical_tail_aspect_ratio'       ,    'vehicle_configurations.*.wings.vertical_stabilizer.aspect_ratio'     ],
        [ 'design_thrust'                    ,    'vehicle_configurations.*.networks.turbofan.thrust.total_design'   ],
        [ 'MTOW'                             ,   ['vehicle_configurations.*.mass_properties.takeoff'   ,
                                                  'vehicle_configurations.*.mass_properties.max_takeoff'               ]],
        [ 'design_TOW'                       ,    'vehicle_configurations.base.mass_properties.takeoff'                 ],
        [ 'short_field_TOW'                  ,    'vehicle_configurations.short_field_takeoff.mass_properties.takeoff'  ],
        [ 'flap_takeoff_angle'               ,   ['vehicle_configurations.takeoff.wings.main_wing.control_surfaces.flap.deflection',
                                                  'vehicle_configurations.short_field_takeoff.wings.main_wing.control_surfaces.flap.deflection']],
        [ 'flap_landing_angle'               ,    'vehicle_configurations.landing.wings.main_wing.control_surfaces.flap.deflection'          ],
        [ 'slat_takeoff_angle'               ,   ['vehicle_configurations.takeoff.wings.main_wing.control_surfaces.slat.deflection',
                                                  'vehicle_configurations.short_field_takeoff.wings.main_wing.control_surfaces.slat.deflection']],
        [ 'slat_landing_angle'               ,    'vehicle_configurations.landing.wings.main_wing.control_surfaces.slat.deflection'          ],
        [ 'wing_span'                        ,    'vehicle_configurations.base.wings.main_wing.spans.projected'         ],
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
    vehicle = vehicle_setup()
    nexus.vehicle_configurations = configs_setup(vehicle)
    

    # -------------------------------------------------------------------
    #  Analyses
    # -------------------------------------------------------------------
    nexus.analyses = Noise_Analyses.setup(nexus.vehicle_configurations)


    # -------------------------------------------------------------------
    #  Missions
    # -------------------------------------------------------------------
    nexus.missions = Noise_Missions.setup(nexus.analyses)

    # -------------------------------------------------------------------
    #  New Regression Flag
    # -------------------------------------------------------------------
    nexus.save_data = generate_new_truth_data

    # -------------------------------------------------------------------
    #  Procedure
    # -------------------------------------------------------------------
    nexus.procedure = Noise_Procedure.setup()

    # -------------------------------------------------------------------
    #  Summary
    # -------------------------------------------------------------------
    nexus.summary = Data()

    return nexus


if __name__ == '__main__':
    main()
