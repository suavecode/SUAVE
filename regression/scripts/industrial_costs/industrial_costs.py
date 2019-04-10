# industrial_costs.py
#
# Created:  Sep 2016, T. Orra

""" regression for industrial costs methods
"""
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units
from SUAVE.Core import Data

import numpy as np

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():

    # list of airplanes to compute manufacturing costs
    config_list = ['B747','B777-200','A380','L-500','MRJ-90','E170-AR','E190-AR','ERJ-145','A321-200','B737-900ER','A330-300','A350-900','B787-8']

    # outputs list
    output_list = np.zeros_like([config_list,config_list])

    # loop to compute industrial cost of each airplane
    for idx,item in enumerate(config_list):
        config = define_config(item)
        costs         = SUAVE.Analyses.Costs.Costs()
        costs.vehicle = config
        costs.evaluate()

        nrec = (config.costs.industrial.non_recurring.total - config.costs.industrial.non_recurring.breakdown.tooling_production) / 1e6
        rec  = config.costs.industrial.unit_cost / 1e6

        print('{:10s} => NREC: {:10.2f} , REC: {:7.2f}'.format(item,nrec,rec))

        output_list[:,idx] = nrec,rec

    print('')
    test = check_results(config_list,output_list)

    return output_list

def define_config(tag):

    config         = SUAVE.Vehicle()
    config.tag     = tag
    gt_engine      = SUAVE.Components.Energy.Networks.Turbofan()
    gt_engine.tag  = 'turbofan'
    manufact_costs = config.costs.industrial
# ===================================
    if tag == 'B777-200':
# ===================================
# B777 - 200
# ===================================
        gt_engine.number_of_engines                 = 2
        gt_engine.sealevel_static_thrust            = 110000 * Units.lbf
        config.append_component(gt_engine)
        config.mass_properties.empty                 = 326000 * Units.lb
        config.envelope.maximum_mach_operational     = 0.89
        config.passengers                            = 250

        manufact_costs.avionics_cost                 = 250000.
        manufact_costs.production_total_units        = 500
        manufact_costs.units_to_amortize             = 500
        manufact_costs.prototypes_units              = 9
        manufact_costs.reference_year                = 2004

        manufact_costs.difficulty_factor = 1.5          # (1.0 for conventional, 1.5 for moderately advanc. tech., 2 for agressive use of adv. tech.)
        manufact_costs.cad_factor        = 1.2          # (1.2 for learning, 1.0 for manual, 0.8 for experienced)
        manufact_costs.stealth           = 0.0          # (0 for non-stealth, 1 for stealth)
        manufact_costs.material_factor   = 1.0          # (1 for conventional Al, 1.5 for stainless steel, 2~2.5 for composites, 3 for carbon fiber)
        manufact_costs.aircraft_type     = 'commercial' # ('military','general aviation','regional','commercial','business')

# ===================================
    elif tag == 'E175-E2':
# ===================================
# E175-E2
# ===================================
        gt_engine.number_of_engines                 = 2
        gt_engine.sealevel_static_thrust            = 15000 * Units.lbf
        config.append_component(gt_engine)
        config.mass_properties.empty                 = 25500.
        config.envelope.maximum_mach_operational     = 0.82
        config.passengers                            = 80

        manufact_costs.avionics_cost                 = 1000000.
        manufact_costs.production_total_units        = 1000
        manufact_costs.units_to_amortize             = 1000
        manufact_costs.prototypes_units              = 4
        manufact_costs.reference_year                = 2010

        manufact_costs.difficulty_factor = 1.0         # (1.0 for conventional, 1.5 for moderately advanc. tech., 2 for agressive use of adv. tech.)
        manufact_costs.cad_factor        = 0.8         # (1.2 for learning, 1.0 for manual, 0.8 for experienced)
        manufact_costs.stealth           = 0.0         # (0 for non-stealth, 1 for stealth)
        manufact_costs.material_factor   = 1.0         # (1 for conventional Al, 1.5 for stainless steel, 2~2.5 for composites, 3 for carbon fiber)
        manufact_costs.aircraft_type     = 'regional'  # ('military','general aviation','regional','commercial','business')

# ===================================
    elif tag == 'E190-E2':
# ===================================
# E190-E2
# ===================================
        gt_engine.number_of_engines                 = 2
        gt_engine.sealevel_static_thrust            = 19000 * Units.lbf
        config.append_component(gt_engine)
        config.mass_properties.empty                 = 33000.
        config.envelope.maximum_mach_operational     = 0.82
        config.passengers                            = 110

        manufact_costs.avionics_cost                 = 1000000.
        manufact_costs.production_total_units        = 1000
        manufact_costs.units_to_amortize             = 1000
        manufact_costs.prototypes_units              = 4
        manufact_costs.reference_year                = 2010

        manufact_costs.difficulty_factor = 1.0         # (1.0 for conventional, 1.5 for moderately advanc. tech., 2 for agressive use of adv. tech.)
        manufact_costs.cad_factor        = 0.8         # (1.2 for learning, 1.0 for manual, 0.8 for experienced)
        manufact_costs.stealth           = 0.0         # (0 for non-stealth, 1 for stealth)
        manufact_costs.material_factor   = 1.0         # (1 for conventional Al, 1.5 for stainless steel, 2~2.5 for composites, 3 for carbon fiber)
        manufact_costs.aircraft_type     = 'regional'  # ('military','general aviation','regional','commercial','business')

# ===================================
    elif tag == 'E195-E2':
# ===================================
# E195-E2
# ===================================
        gt_engine.number_of_engines                 = 2
        gt_engine.sealevel_static_thrust            = 23000 * Units.lbf
        config.append_component(gt_engine)
        config.mass_properties.empty                 = 36100.
        config.envelope.maximum_mach_operational     = 0.82
        config.passengers                            = 122

        manufact_costs.avionics_cost                 = 1000000.
        manufact_costs.production_total_units        = 1000
        manufact_costs.units_to_amortize             = 1000
        manufact_costs.prototypes_units              = 4
        manufact_costs.reference_year                = 2010

        manufact_costs.difficulty_factor = 1.0         # (1.0 for conventional, 1.5 for moderately advanc. tech., 2 for agressive use of adv. tech.)
        manufact_costs.cad_factor        = 0.8         # (1.2 for learning, 1.0 for manual, 0.8 for experienced)
        manufact_costs.stealth           = 0.0         # (0 for non-stealth, 1 for stealth)
        manufact_costs.material_factor   = 1.0         # (1 for conventional Al, 1.5 for stainless steel, 2~2.5 for composites, 3 for carbon fiber)
        manufact_costs.aircraft_type     = 'regional'  # ('military','general aviation','regional','commercial','business')

# ===================================
    elif tag == 'E190-AR':
# ===================================
# E190-AR
# ===================================
        gt_engine.number_of_engines                 = 2
        gt_engine.sealevel_static_thrust            = 20000 * Units.lbf
        config.append_component(gt_engine)
        config.mass_properties.empty                 = 28080.
        config.envelope.maximum_mach_operational     = 0.82
        config.passengers                            = 100

        manufact_costs.avionics_cost                 = 1000000.
        manufact_costs.production_total_units        = 1000
        manufact_costs.units_to_amortize             = 1000
        manufact_costs.prototypes_units              = 6
        manufact_costs.reference_year                = 1999

        manufact_costs.difficulty_factor = 1.0         # (1.0 for conventional, 1.5 for moderately advanc. tech., 2 for agressive use of adv. tech.)
        manufact_costs.cad_factor        = 1.0         # (1.2 for learning, 1.0 for manual, 0.8 for experienced)
        manufact_costs.stealth           = 0.0         # (0 for non-stealth, 1 for stealth)
        manufact_costs.material_factor   = 1.0         # (1 for conventional Al, 1.5 for stainless steel, 2~2.5 for composites, 3 for carbon fiber)
        manufact_costs.aircraft_type     = 'regional'  # ('military','general aviation','regional','commercial','business')

# ===================================
    elif tag == 'E170-AR':
# ===================================
# E170-AR
# ===================================
        gt_engine.number_of_engines                  = 2
        gt_engine.sealevel_static_thrust             = 14200 * Units.lbf
        config.append_component(gt_engine)
        config.mass_properties.empty                 = 21140.
        config.envelope.maximum_mach_operational     = 0.82
        config.passengers                            = 80

        manufact_costs.avionics_cost                 = 1000000.
        manufact_costs.production_total_units        = 1000
        manufact_costs.units_to_amortize             = 1000
        manufact_costs.prototypes_units              = 6
        manufact_costs.reference_year                = 1999

        manufact_costs.difficulty_factor = 1.0         # (1.0 for conventional, 1.5 for moderately advanc. tech., 2 for agressive use of adv. tech.)
        manufact_costs.cad_factor        = 1.0         # (1.2 for learning, 1.0 for manual, 0.8 for experienced)
        manufact_costs.stealth           = 0.0         # (0 for non-stealth, 1 for stealth)
        manufact_costs.material_factor   = 1.0         # (1 for conventional Al, 1.5 for stainless steel, 2~2.5 for composites, 3 for carbon fiber)
        manufact_costs.aircraft_type     = 'regional'  # ('military','general aviation','regional','commercial','business')

# ===================================
    elif tag == 'A380':
# ===================================
# A380
# ===================================
        gt_engine.number_of_engines                  = 4
        gt_engine.sealevel_static_thrust             = 70000. * Units.lbf
        config.append_component(gt_engine)
        config.mass_properties.empty                 = 610200. * Units.lb
        config.envelope.maximum_mach_operational     = 0.96
        config.passengers                            = 860

        manufact_costs.avionics_cost                 = 5000000.
        manufact_costs.production_total_units        = 500
        manufact_costs.units_to_amortize             = 500
        manufact_costs.prototypes_units              = 9
        manufact_costs.reference_year                = 2004

        manufact_costs.difficulty_factor = 2.0           # (1.0 for conventional, 1.5 for moderately advanc. tech., 2 for agressive use of adv. tech.)
        manufact_costs.cad_factor        = 0.8           # (1.2 for learning, 1.0 for manual, 0.8 for experienced)
        manufact_costs.stealth           = 0.0           # (0 for non-stealth, 1 for stealth)
        manufact_costs.material_factor   = 1.5           # (1 for conventional Al, 1.5 for stainless steel, 2~2.5 for composites, 3 for carbon fiber)
        manufact_costs.aircraft_type     = 'commercial'  # ('military','general aviation','regional','commercial','business')

# ===================================
    elif tag == 'B747':
# ===================================
# B747
# ===================================
        gt_engine.number_of_engines                  = 4
        gt_engine.sealevel_static_thrust             = 50000. * Units.lbf
        config.append_component(gt_engine)
        config.mass_properties.empty                 = 360000. * Units.lb
        config.envelope.maximum_mach_operational     = 0.89
        config.passengers                            = 480

        manufact_costs.avionics_cost                 = 5000000.
        manufact_costs.production_total_units        = 500
        manufact_costs.units_to_amortize             = 500
        manufact_costs.prototypes_units              = 5
        manufact_costs.reference_year                = 2004

        manufact_costs.difficulty_factor = 1.25           # (1.0 for conventional, 1.5 for moderately advanc. tech., 2 for agressive use of adv. tech.)
        manufact_costs.cad_factor        = 1.0           # (1.2 for learning, 1.0 for manual, 0.8 for experienced)
        manufact_costs.stealth           = 0.0           # (0 for non-stealth, 1 for stealth)
        manufact_costs.material_factor   = 1.0           # (1 for conventional Al, 1.5 for stainless steel, 2~2.5 for composites, 3 for carbon fiber)
        manufact_costs.aircraft_type     = 'commercial'  # ('military','general aviation','regional','commercial','business')

# ===================================
    elif tag == 'ERJ-145':
# ===================================
# ERJ-145
# ===================================
        gt_engine.number_of_engines                  = 2
        gt_engine.sealevel_static_thrust             = 7800. * Units.lbf
        config.append_component(gt_engine)
        config.mass_properties.empty                 = 11800.
        config.envelope.maximum_mach_operational     = 0.78
        config.passengers                            = 44

        manufact_costs.avionics_cost                 = 500000.
        manufact_costs.production_total_units        = 700
        manufact_costs.units_to_amortize             = 700
        manufact_costs.prototypes_units              = 3
        manufact_costs.reference_year                = 1993

        manufact_costs.difficulty_factor = 1.5           # (1.0 for conventional, 1.5 for moderately advanc. tech., 2 for agressive use of adv. tech.)
        manufact_costs.cad_factor        = 1.0           # (1.2 for learning, 1.0 for manual, 0.8 for experienced)
        manufact_costs.stealth           = 0.0           # (0 for non-stealth, 1 for stealth)
        manufact_costs.material_factor   = 1.0           # (1 for conventional Al, 1.5 for stainless steel, 2~2.5 for composites, 3 for carbon fiber)
        manufact_costs.aircraft_type     = 'regional'  # ('military','general aviation','regional','commercial','business')

# ===================================
    elif tag == 'MRJ-70':
# ===================================
# MRJ70
# ===================================
        gt_engine.number_of_engines                  = 2
        gt_engine.sealevel_static_thrust             = 15600. * Units.lbf
        config.append_component(gt_engine)
        config.mass_properties.empty                 = 20000.
        config.envelope.maximum_mach_operational     = 0.78
        config.passengers                            = 80

        manufact_costs.avionics_cost                 = 1000000.
        manufact_costs.production_total_units        = 700
        manufact_costs.units_to_amortize             = 700
        manufact_costs.prototypes_units              = 4
        manufact_costs.reference_year                = 2010

        manufact_costs.difficulty_factor = 1.5           # (1.0 for conventional, 1.5 for moderately advanc. tech., 2 for agressive use of adv. tech.)
        manufact_costs.cad_factor        = 1.2           # (1.2 for learning, 1.0 for manual, 0.8 for experienced)
        manufact_costs.stealth           = 0.0           # (0 for non-stealth, 1 for stealth)
        manufact_costs.material_factor   = 1.0           # (1 for conventional Al, 1.5 for stainless steel, 2~2.5 for composites, 3 for carbon fiber)
        manufact_costs.aircraft_type     = 'regional'  # ('military','general aviation','regional','commercial','business')

# ===================================
    elif tag == 'MRJ-90':
# ===================================
# MRJ-90
# ===================================
        gt_engine.number_of_engines                  = 2
        gt_engine.sealevel_static_thrust             = 15600. * Units.lbf
        config.append_component(gt_engine)
        config.mass_properties.empty                 = 26000.
        config.envelope.maximum_mach_operational     = 0.78
        config.passengers                            = 92

        manufact_costs.avionics_cost                 = 1000000.
        manufact_costs.production_total_units        = 700
        manufact_costs.units_to_amortize             = 700
        manufact_costs.prototypes_units              = 4
        manufact_costs.reference_year                = 2010

        manufact_costs.difficulty_factor = 1.5           # (1.0 for conventional, 1.5 for moderately advanc. tech., 2 for agressive use of adv. tech.)
        manufact_costs.cad_factor        = 1.2           # (1.2 for learning, 1.0 for manual, 0.8 for experienced)
        manufact_costs.stealth           = 0.0           # (0 for non-stealth, 1 for stealth)
        manufact_costs.material_factor   = 1.0           # (1 for conventional Al, 1.5 for stainless steel, 2~2.5 for composites, 3 for carbon fiber)
        manufact_costs.aircraft_type     = 'regional'  # ('military','general aviation','regional','commercial','business')

# ===================================
    elif tag == 'L-500':
# ===================================
# L500
# ===================================
        gt_engine.number_of_engines                  = 2
        gt_engine.sealevel_static_thrust             = 7000. * Units.lbf
        config.append_component(gt_engine)
        config.mass_properties.empty                 = 10600.
        config.envelope.maximum_mach_operational     = 0.83
        config.passengers                            = 12

        manufact_costs.avionics_cost                 = 1000000.
        manufact_costs.production_total_units        = 700
        manufact_costs.units_to_amortize             = 700
        manufact_costs.prototypes_units              = 4
        manufact_costs.reference_year                = 2008

        manufact_costs.difficulty_factor = 1.5          # (1.0 for conventional, 1.5 for moderately advanc. tech., 2 for agressive use of adv. tech.)
        manufact_costs.cad_factor        = 0.8           # (1.2 for learning, 1.0 for manual, 0.8 for experienced)
        manufact_costs.stealth           = 0.0           # (0 for non-stealth, 1 for stealth)
        manufact_costs.material_factor   = 1.0           # (1 for conventional Al, 1.5 for stainless steel, 2~2.5 for composites, 3 for carbon fiber)
        manufact_costs.aircraft_type     = 'business'  # ('military','general aviation','regional','commercial','business')

# ===================================
    elif tag == 'A321-200':
# ===================================
# A321-200
# ===================================
        gt_engine.number_of_engines                  = 2
        gt_engine.sealevel_static_thrust             = 33000. * Units.lbf
        config.append_component(gt_engine)
        config.mass_properties.empty                 = 48500.
        config.envelope.maximum_mach_operational     = 0.82
        config.passengers                            = 236

        manufact_costs.avionics_cost                 = 2000000.
        manufact_costs.production_total_units        = 700
        manufact_costs.units_to_amortize             = 700
        manufact_costs.prototypes_units              = 6
        manufact_costs.reference_year                = 2005

        manufact_costs.difficulty_factor = 1.0           # (1.0 for conventional, 1.5 for moderately advanc. tech., 2 for agressive use of adv. tech.)
        manufact_costs.cad_factor        = 0.8           # (1.2 for learning, 1.0 for manual, 0.8 for experienced)
        manufact_costs.stealth           = 0.0           # (0 for non-stealth, 1 for stealth)
        manufact_costs.material_factor   = 1.0           # (1 for conventional Al, 1.5 for stainless steel, 2~2.5 for composites, 3 for carbon fiber)
        manufact_costs.aircraft_type     = 'commercial'  # ('military','general aviation','regional','commercial','business')

# ===================================
    elif tag == 'B737-900ER':
# ===================================
# B737-900ER
# ===================================
        gt_engine.number_of_engines                  = 2
        gt_engine.sealevel_static_thrust             = 27300. * Units.lbf
        config.append_component(gt_engine)
        config.mass_properties.empty                 = 44676.
        config.envelope.maximum_mach_operational     = 0.78
        config.passengers                            = 220

        manufact_costs.avionics_cost                 = 2000000.
        manufact_costs.production_total_units        = 700
        manufact_costs.units_to_amortize             = 700
        manufact_costs.prototypes_units              = 6
        manufact_costs.reference_year                = 2005

        manufact_costs.difficulty_factor = 1.0           # (1.0 for conventional, 1.5 for moderately advanc. tech., 2 for agressive use of adv. tech.)
        manufact_costs.cad_factor        = 0.8           # (1.2 for learning, 1.0 for manual, 0.8 for experienced)
        manufact_costs.stealth           = 0.0           # (0 for non-stealth, 1 for stealth)
        manufact_costs.material_factor   = 1.0           # (1 for conventional Al, 1.5 for stainless steel, 2~2.5 for composites, 3 for carbon fiber)
        manufact_costs.aircraft_type     = 'commercial'  # ('military','general aviation','regional','commercial','business')


# ===================================
    elif tag == 'A330-300':
# ===================================
# A330-300
# ===================================
        gt_engine.number_of_engines                  = 2
        gt_engine.sealevel_static_thrust             = 70000. * Units.lbf
        config.append_component(gt_engine)
        config.mass_properties.empty                 = 122780.
        config.envelope.maximum_mach_operational     = 0.86
        config.passengers                            = 300

        manufact_costs.avionics_cost                 = 2000000.
        manufact_costs.production_total_units        = 700
        manufact_costs.units_to_amortize             = 700
        manufact_costs.prototypes_units              = 6
        manufact_costs.reference_year                = 2005

        manufact_costs.difficulty_factor = 1.0           # (1.0 for conventional, 1.5 for moderately advanc. tech., 2 for agressive use of adv. tech.)
        manufact_costs.cad_factor        = 0.8           # (1.2 for learning, 1.0 for manual, 0.8 for experienced)
        manufact_costs.stealth           = 0.0           # (0 for non-stealth, 1 for stealth)
        manufact_costs.material_factor   = 1.0           # (1 for conventional Al, 1.5 for stainless steel, 2~2.5 for composites, 3 for carbon fiber)
        manufact_costs.aircraft_type     = 'commercial'  # ('military','general aviation','regional','commercial','business')

# ===================================
    elif tag == 'A350-900':
# ===================================
# A350-900
# ===================================
        gt_engine.number_of_engines                  = 2
        gt_engine.sealevel_static_thrust             = 84200. * Units.lbf
        config.append_component(gt_engine)
        config.mass_properties.empty                 = 145000.
        config.envelope.maximum_mach_operational     = 0.89
        config.passengers                            = 325

        manufact_costs.avionics_cost                 = 2000000.
        manufact_costs.production_total_units        = 2000
        manufact_costs.units_to_amortize             = 2000
        manufact_costs.prototypes_units              = 6
        manufact_costs.reference_year                = 2005

        manufact_costs.difficulty_factor = 1.5           # (1.0 for conventional, 1.5 for moderately advanc. tech., 2 for agressive use of adv. tech.)
        manufact_costs.cad_factor        = 0.8           # (1.2 for learning, 1.0 for manual, 0.8 for experienced)
        manufact_costs.stealth           = 0.0           # (0 for non-stealth, 1 for stealth)
        manufact_costs.material_factor   = 1.5           # (1 for conventional Al, 1.5 for stainless steel, 2~2.5 for composites, 3 for carbon fiber)
        manufact_costs.aircraft_type     = 'commercial'  # ('military','general aviation','regional','commercial','business')

# ===================================
    elif tag == 'B787-8':
# ===================================
# B787-8
# ===================================
        gt_engine.number_of_engines                  = 2
        gt_engine.sealevel_static_thrust             = 64000. * Units.lbf
        config.append_component(gt_engine)
        config.mass_properties.empty                 = 120000.
        config.envelope.maximum_mach_operational     = 0.89
        config.passengers                            = 359

        manufact_costs.avionics_cost                 = 2000000.
        manufact_costs.production_total_units        = 2000
        manufact_costs.units_to_amortize             = 2000
        manufact_costs.prototypes_units              = 6
        manufact_costs.reference_year                = 2005

        manufact_costs.difficulty_factor = 1.5           # (1.0 for conventional, 1.5 for moderately advanc. tech., 2 for agressive use of adv. tech.)
        manufact_costs.cad_factor        = 0.8           # (1.2 for learning, 1.0 for manual, 0.8 for experienced)
        manufact_costs.stealth           = 0.0           # (0 for non-stealth, 1 for stealth)
        manufact_costs.material_factor   = 1.5           # (1 for conventional Al, 1.5 for stainless steel, 2~2.5 for composites, 3 for carbon fiber)
        manufact_costs.aircraft_type     = 'commercial'  # ('military','general aviation','regional','commercial','business')


    return config
#==================================

def check_results(config_list,new_results):

    # check segment values
    old_results = [['4474.11819', '6501.93808', '14458.8219', '592.762814',
        '1231.10138', '662.096986', '823.261765', '366.953003',
        '1434.72752', '1290.42378', '3038.82443', '5283.75084',
        '4553.70202'],
       ['106.368618', '108.551707', '228.829859', '14.0782491',
        '27.1352858', '13.3878321', '16.5547165', '9.65499519',
        '32.8083219', '29.6803338', '62.9236325', '69.9310621',
        '60.1980080']]

    # do the check
    for idx,item in enumerate(config_list):
        print(item)

        old_val = float(old_results[0][idx])
        new_val = float(new_results[0][idx])
        err = (new_val-old_val)/old_val
        print('Error at NREC:' , err)
        assert np.abs(err) < 1e-6 , 'NREC Cost Failed : %s' % item

        old_val = float(old_results[1][idx])
        new_val = float(new_results[1][idx])
        err = (new_val-old_val)/old_val
        print('Error at REC:' , err)
        assert np.abs(err) < 1e-6 , 'REC Cost Failed : %s' % item

        print('')

    return

#==================================
if __name__ == '__main__':
    output = main()
