# weights.py
# Created:
# Modified: Mar 2020, M. Clarke

import SUAVE
import numpy as np
from SUAVE.Core import Units
from SUAVE.Methods.Weights.Correlations import Propulsion as Propulsion
from SUAVE.Methods.Weights.Correlations import Transport as Transport
from SUAVE.Methods.Weights.Correlations import Common as Common
from SUAVE.Methods.Weights.Correlations import General_Aviation as General_Aviation
from SUAVE.Methods.Weights.Correlations import BWB as BWB
from SUAVE.Methods.Weights.Correlations import Human_Powered as HP

from SUAVE.Core import (Data, Container,)
from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing

import sys

sys.path.append('../Vehicles')
# the analysis functions

from Boeing_737 import vehicle_setup
from Cessna_172 import vehicle_setup as vehicle_setup_general_aviation
from BWB import vehicle_setup  as bwb_setup
from Solar_UAV import vehicle_setup  as hp_setup


def main():

    vehicle = vehicle_setup()
    weight = Common.empty_weight(vehicle, method_type = "SUAVE")

    # regression values SUAVE
    actual = Data()
    actual.payload         = 27349.9081525      # includes cargo #17349.9081525 #without cargo
    actual.pax             = 15036.587065500002
    actual.bag             = 2313.3210870000003
    actual.fuel            = 8964.067406365866  # includes cargo #22177.6377131 #without cargo
    actual.empty           = 39739.86626503414
    actual.wing            = 8688.245864290779
    actual.fuselage        = 6612.201567847215
    actual.propulsion      = 6838.185174956626
    actual.landing_gear    = 3160.632
    actual.systems         = 11478.036709368022
    actual.wt_furnish      = 6431.803728889001
    actual.horizontal_tail = 1886.1811683764736
    actual.vertical_tail   = 1076.3837801950292

    # error calculations
    error                 = Data()
    error.payload         = (actual.payload - weight.payload_breakdown.total)/actual.payload
    error.pax             = (actual.pax - weight.payload_breakdown.passengers)/actual.pax
    error.bag             = (actual.bag - weight.payload_breakdown.baggage)/actual.bag
    error.fuel            = (actual.fuel - weight.fuel)/actual.fuel
    error.empty           = (actual.empty - weight.empty)/actual.empty
    error.wing            = (actual.wing - weight.structures.wing)/actual.wing
    error.fuselage        = (actual.fuselage - weight.structures.fuselage)/actual.fuselage
    error.propulsion      = (actual.propulsion - weight.propulsion_breakdown.total)/actual.propulsion
    error.landing_gear    = (actual.landing_gear - weight.structures.main_landing_gear
                             - weight.structures.nose_landing_gear)/actual.landing_gear
    error.systems         = (actual.systems - weight.systems_breakdown.total)/actual.systems
    error.wt_furnish      = (actual.wt_furnish - weight.systems_breakdown.furnish)/actual.wt_furnish
    error.horizontal_tail = (actual.horizontal_tail - weight.structures.horizontal_tail)/actual.horizontal_tail
    error.vertical_tail   = (actual.vertical_tail - weight.structures.vertical_tail)/actual.vertical_tail

    print('Results (kg)')
    print(weight)

    print('Relative Errors')
    print(error)

    for k, v in error.items():
        assert (np.abs(v) < 1E-6)

    weight = Common.empty_weight(vehicle, method_type="FLOPS Simple")

    # regression values FLOPS Complex
    actual = Data()
    actual.payload = 19164.047296928187  # includes cargo #17349.9081525 #without cargo
    actual.pax = 5771.176369328185
    actual.bag = 3392.8709276
    actual.fuel = 20031.585931973066  # includes cargo #22177.6377131 #without cargo
    actual.empty = 35276.43161054148
    actual.wing = 6129.985979314519
    actual.fuselage = 7304.86777971127
    actual.propulsion = 6158.342445321374
    actual.landing_gear = 2695.109360215617 + 335.5055180687866
    actual.systems = 10968.186283406896
    actual.wt_furnish = 6453.793837053036
    actual.horizontal_tail = 657.5705301445911
    actual.vertical_tail = 509.7069372086644

    # error calculations
    error = Data()
    error.payload = (actual.payload - weight.payload_breakdown.total) / actual.payload
    error.pax = (actual.pax - weight.payload_breakdown.passengers) / actual.pax
    error.bag = (actual.bag - weight.payload_breakdown.baggage) / actual.bag
    error.fuel = (actual.fuel - weight.fuel) / actual.fuel
    error.empty = (actual.empty - weight.empty) / actual.empty
    error.wing = (actual.wing - weight.structures.wing) / actual.wing
    error.fuselage = (actual.fuselage - weight.structures.fuselage) / actual.fuselage
    error.propulsion = (actual.propulsion - weight.propulsion_breakdown.total) / actual.propulsion
    error.landing_gear = (actual.landing_gear - weight.structures.main_landing_gear
                          - weight.structures.nose_landing_gear) / actual.landing_gear
    error.systems = (actual.systems - weight.systems_breakdown.total) / actual.systems
    error.wt_furnish = (actual.wt_furnish - weight.systems_breakdown.furnish) / actual.wt_furnish
    error.horizontal_tail = (actual.horizontal_tail - weight.structures.horizontal_tail) / actual.horizontal_tail
    error.vertical_tail = (actual.vertical_tail - weight.structures.vertical_tail) / actual.vertical_tail

    print('Results (kg)')
    print(weight)

    print('Relative Errors')
    print(error)

    for k, v in error.items():
        assert (np.abs(v) < 1E-6)

    weight = Common.empty_weight(vehicle, method_type="FLOPS Complex")

    # regression values FLOPS Complex
    actual                  = Data()
    actual.payload          = 19164.047296928187  # includes cargo #17349.9081525 #without cargo
    actual.pax              = 5771.176369328185
    actual.bag              = 3392.8709276
    actual.fuel             = 19123.46037492172  # includes cargo #22177.6377131 #without cargo
    actual.empty            = 36184.55716759282
    actual.wing             = 7038.111536365861
    actual.fuselage         = 7304.86777971127
    actual.propulsion       = 6158.342445321374
    actual.landing_gear     = 2695.109360215617 + 335.5055180687866
    actual.systems          = 10968.186283406896
    actual.wt_furnish       = 6453.793837053036
    actual.horizontal_tail  = 657.5705301445911
    actual.vertical_tail    = 509.7069372086644

    # error calculations
    error                       = Data()
    error.payload               = (actual.payload - weight.payload_breakdown.total) / actual.payload
    error.pax                   = (actual.pax - weight.payload_breakdown.passengers) / actual.pax
    error.bag                   = (actual.bag - weight.payload_breakdown.baggage) / actual.bag
    error.fuel                  = (actual.fuel - weight.fuel) / actual.fuel
    error.empty                 = (actual.empty - weight.empty) / actual.empty
    error.wing                  = (actual.wing - weight.structures.wing) / actual.wing
    error.fuselage              = (actual.fuselage - weight.structures.fuselage) / actual.fuselage
    error.propulsion            = (actual.propulsion - weight.propulsion_breakdown.total) / actual.propulsion
    error.landing_gear          = (actual.landing_gear - weight.structures.main_landing_gear
                                - weight.structures.nose_landing_gear) / actual.landing_gear
    error.systems               = (actual.systems - weight.systems_breakdown.total) / actual.systems
    error.wt_furnish            = (actual.wt_furnish - weight.systems_breakdown.furnish) / actual.wt_furnish
    error.horizontal_tail       = (actual.horizontal_tail - weight.structures.horizontal_tail) / actual.horizontal_tail
    error.vertical_tail         = (actual.vertical_tail - weight.structures.vertical_tail) / actual.vertical_tail

    print('Results (kg)')
    print(weight)

    print('Relative Errors')
    print(error)

    for k, v in error.items():
        assert (np.abs(v) < 1E-6)

    weight = Common.empty_weight(vehicle, method_type="Raymer")

    # regression values Raymer
    actual                  = Data()
    actual.payload          = 27349.9081525  # includes cargo #17349.9081525 #without cargo
    actual.pax              = 15036.587065500002
    actual.bag              = 2313.3210870000003
    actual.fuel             = 15291.689207249146  # includes cargo #22177.6377131 #without cargo
    actual.empty            = 33412.244464150855
    actual.wing             = 6268.223851650347
    actual.fuselage         = 6655.851643473622
    actual.propulsion       = 5444.68675310034
    actual.landing_gear     = 2359.709779278413 + 427.45355397385555
    actual.systems          = 10121.900868622723
    actual.wt_furnish       = 5810.9423987753435
    actual.horizontal_tail  = 766.8742732883632
    actual.vertical_tail    = 713.7539324202631

    # error calculations
    error                   = Data()
    error.payload           = (actual.payload - weight.payload_breakdown.total) / actual.payload
    error.pax               = (actual.pax - weight.payload_breakdown.passengers) / actual.pax
    error.bag               = (actual.bag - weight.payload_breakdown.baggage) / actual.bag
    error.fuel              = (actual.fuel - weight.fuel) / actual.fuel
    error.empty             = (actual.empty - weight.empty) / actual.empty
    error.wing              = (actual.wing - weight.structures.wing) / actual.wing
    error.fuselage          = (actual.fuselage - weight.structures.fuselage) / actual.fuselage
    error.propulsion        = (actual.propulsion - weight.propulsion_breakdown.total) / actual.propulsion
    error.landing_gear      = (actual.landing_gear - weight.structures.main_landing_gear
                            - weight.structures.nose_landing_gear) / actual.landing_gear
    error.systems           = (actual.systems - weight.systems_breakdown.total) / actual.systems
    error.wt_furnish        = (actual.wt_furnish - weight.systems_breakdown.furnish) / actual.wt_furnish
    error.horizontal_tail   = (actual.horizontal_tail - weight.structures.horizontal_tail) / actual.horizontal_tail
    error.vertical_tail     = (actual.vertical_tail - weight.structures.vertical_tail) / actual.vertical_tail

    print('Results (kg)')
    print(weight)

    print('Relative Errors')
    print(error)

    for k, v in error.items():
        assert (np.abs(v) < 1E-6)

    #General Aviation weights; note that values are taken from Raymer,
    #but there is a huge spread among the GA designs, so individual components
    #differ a good deal from the actual design

    vehicle        = vehicle_setup_general_aviation()
    weight         = General_Aviation.empty(vehicle)
    weight.fuel    = vehicle.fuel.mass_properties.mass
    actual         = Data()
    actual.bag     = 0.
    actual.empty   = 700.0097482541994
    actual.fuel    = 48.417662245800784

    actual.wing            = 152.25407206578896
    actual.fuselage        = 126.7421108234472
    actual.propulsion      = 224.40728553408732
    actual.landing_gear    = 67.81320006645151
    actual.furnishing      = 37.8341395817
    actual.electrical      = 41.28649399649684
    actual.control_systems = 20.51671046011007
    actual.fuel_systems    = 20.173688786768366
    actual.systems         = 102.62736387596043

    error                 = Data()
    error.fuel            = (actual.fuel - weight.fuel)/actual.fuel
    error.empty           = (actual.empty - weight.empty)/actual.empty
    error.wing            = (actual.wing - weight.structures.wing)/actual.wing
    error.fuselage        = (actual.fuselage - weight.structures.fuselage)/actual.fuselage
    error.propulsion      = (actual.propulsion - weight.propulsion_breakdown.total)/actual.propulsion
    error.landing_gear    = (actual.landing_gear - (weight.structures.main_landing_gear+weight.structures.nose_landing_gear))/actual.landing_gear
    error.furnishing      = (actual.furnishing-weight.systems_breakdown.furnish)/actual.furnishing
    error.electrical      = (actual.electrical-weight.systems_breakdown.electrical)/actual.electrical
    error.control_systems = (actual.control_systems-weight.systems_breakdown.control_systems)/actual.control_systems
    error.fuel_systems    = (actual.fuel_systems-weight.propulsion_breakdown.fuel_system)/actual.fuel_systems
    error.systems         = (actual.systems - weight.systems_breakdown.total)/actual.systems

    print('Results (kg)')
    print(weight)

    print('Relative Errors')
    print(error)

    for k, v in error.items():
        assert (np.abs(v) < 1E-6)

    # BWB WEIGHTS
    vehicle = bwb_setup()
    weight  = BWB.empty(vehicle)

    # regression values
    actual = Data()
    actual.payload         = 27349.9081525 #includes cargo #17349.9081525 #without cargo
    actual.pax             = 15036.587065500002
    actual.bag             = 2313.3210870000003
    actual.fuel            = 23361.42500371662
    actual.empty           = 24417.180232883387
    actual.wing            = 7272.740220314861
    actual.fuselage        = 1.0
    actual.propulsion      = 1413.8593105126783
    actual.landing_gear    = 3160.632
    actual.systems         = 12569.948702055846
    actual.wt_furnish      = 8205.349895589

    # error calculations
    error                 = Data()
    error.payload         = (actual.payload - weight.payload_breakdown.total)/actual.payload
    error.pax             = (actual.pax - weight.payload_breakdown.passengers)/actual.pax
    error.bag             = (actual.bag - weight.payload_breakdown.baggage)/actual.bag
    error.fuel            = (actual.fuel - weight.fuel)/actual.fuel
    error.empty           = (actual.empty - weight.empty)/actual.empty
    error.wing            = (actual.wing - weight.structures.wing)/actual.wing
    error.fuselage        = (actual.fuselage - (weight.structures.fuselage+1.0))/actual.fuselage
    error.propulsion      = (actual.propulsion - weight.propulsion_breakdown.total)/actual.propulsion
    error.systems         = (actual.systems - weight.systems_breakdown.total)/actual.systems
    error.wt_furnish      = (actual.wt_furnish - weight.systems_breakdown.furnish)/actual.wt_furnish

    print('Results (kg)')
    print(weight)

    print('Relative Errors')
    print(error)

    for k, v in error.items():
        assert (np.abs(v) < 1E-6)

    # Human Powered Aircraft
    vehicle = hp_setup()
    weight = HP.empty(vehicle)

    # regression values
    actual = Data()
    actual.empty           = 143.59737768459374
    actual.wing            = 95.43286881794776
    actual.fuselage        = 1.0
    actual.horizontal_tail = 31.749272074174737
    actual.vertical_tail   = 16.415236792471237

    # error calculations
    error                 = Data()
    error.empty           = (actual.empty - weight.empty) / actual.empty
    error.wing            = (actual.wing - weight.wing) / actual.wing
    error.fuselage        = (actual.fuselage - (weight.fuselage + 1.0)) / actual.fuselage
    error.horizontal_tail = (actual.horizontal_tail - weight.horizontal_tail) / actual.horizontal_tail
    error.vertical_tail   = (actual.vertical_tail - weight.vertical_tail) / actual.vertical_tail
    print('Results (kg)')
    print(weight)

    print('Relative Errors')
    print(error)

    for k, v in error.items():
        assert (np.abs(v) < 1E-6)



    return

# ----------------------------------------------------------------------
#   Call Main
# ----------------------------------------------------------------------

if __name__ == '__main__':
    main()
