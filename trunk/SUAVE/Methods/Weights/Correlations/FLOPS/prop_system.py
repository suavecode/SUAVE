## @ingroup Methods-Weights-Correlations-FLOPS
# prop_system.py
#
# Created:  May 2020, W. Van Gijseghem 
# Modified: Oct 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Units, Data
import numpy as np

## @ingroup Methods-Weights-Correlations-FLOPS
def total_prop_flops(vehicle,prop):
    """ Calculate the weight of propulsion system, including:
        - dry engine weight
        - fuel system weight
        - thurst reversers weight
        - electrical system weight
        - starter engine weight
        - nacelle weight
        - cargo containers

        Assumptions:
            1) Rated thrust per scaled engine and rated thurst for baseline are the same
            2) Engine weight scaling parameter is 1.15
            3) Enginge inlet weight scaling exponent is 1
            4) Baseline inlet weight is 0 lbs as in example files FLOPS
            5) Baseline nozzle weight is 0 lbs as in example files FLOPS

        Source:
            The Flight Optimization System Weight Estimation Method

        Inputs:
            vehicle - data dictionary with vehicle properties                   [dimensionless]
                -.design_mach_number: design mach number for cruise flight
                -.mass_properties.max_zero_fuel: zero fuel weight               [kg]
                -.systems.accessories: type of aircraft (short-range, commuter
                                                        medium-range, long-range,
                                                        sst, cargo)
            nacelle - data dictionary with propulsion system properties 
                -.diameter: diameter of nacelle                                 [meters]
                -.length: length of complete engine assembly                    [meters]
            prop.
                -.sealevel_static_thrust: thrust at sea level                   [N]


        Outputs:
            output - data dictionary with weights                               [kilograms]
                    - output.wt_prop: total propulsive system weight
                    - output.wt_thrust_reverser: thurst reverser weight
                    - output.starter: starter engine weight
                    - output.wt_engine_controls: engine controls weight
                    - output.fuel_system: fuel system weight
                    - output.nacelle: nacelle weight
                    - output.wt_eng: dry engine weight

        Properties Used:
            N/A
    """
    
    nacelle_tag     = list(vehicle.nacelles.keys())[0]
    ref_nacelle     = vehicle.nacelles[nacelle_tag]
    NENG            = prop.number_of_engines
    WNAC            = nacelle_FLOPS(prop,ref_nacelle)
    WFSYS           = fuel_system_FLOPS(vehicle, NENG)
    WENG            = engine_FLOPS(vehicle, prop)
    WEC, WSTART     = misc_engine_FLOPS(vehicle,prop,ref_nacelle)
    WTHR            = thrust_reverser_FLOPS(prop)
    WPRO            = NENG * WENG + WFSYS + WEC + WSTART + WTHR + WNAC

    output                      = Data()
    output.wt_prop              = WPRO
    output.wt_thrust_reverser   = WTHR
    output.wt_starter           = WSTART
    output.wt_engine_controls   = WEC
    output.fuel_system          = WFSYS
    output.nacelle              = WNAC
    output.wt_eng               = WENG * NENG
    return output

## @ingroup Methods-Weights-Correlations-FLOPS
def nacelle_FLOPS(prop,nacelle):
    """ Calculates the nacelle weight based on the FLOPS method
    
        Assumptions:
            1) All nacelles are identical
            2) The number of nacelles is the same as the number of engines 

        Source:
            Aircraft Design: A Conceptual Approach

        Inputs:
            prop    - data dictionary for the specific network that is being estimated [dimensionless]
                -.number_of_engines: number of engines
                -.engine_lenght: total length of engine                                  [m]
                -.sealevel_static_thrust: sealevel static thrust of engine               [N]
            nacelle.             
                -.diameter: diameter of nacelle                                          [m]
            WENG    - dry engine weight                                                  [kg]
             
             
        Outputs:             
            WNAC: nacelle weight                                                         [kg]

        Properties Used:
            N/A
    """
      
    NENG   = len(prop.origin)
    TNAC   = NENG + 1. / 2 * (NENG - 2 * np.floor(NENG / 2.))
    DNAC   = nacelle.diameter / Units.ft
    XNAC   = nacelle.length / Units.ft
    FTHRST = prop.sealevel_static_thrust * 1 / Units.lbf
    WNAC   = 0.25 * TNAC * DNAC * XNAC * FTHRST ** 0.36
    return WNAC * Units.lbs

## @ingroup Methods-Weights-Correlations-FLOPS
def thrust_reverser_FLOPS(prop):
    """ Calculates the weight of the thrust reversers of the aircraft
    
        Assumptions:

        Source:
            The Flight Optimization System Weight Estimation Method

        Inputs:
            prop    - data dictionary for the specific network that is being estimated [dimensionless]
                -.number_of_engines: number of engines
                -.sealevel_static_thrust: sealevel static thrust of engine  [N]

        Outputs:
            WTHR: Thrust reversers weight                                   [kg]

        Properties Used:
            N/A
    """
    NENG = prop.number_of_engines
    TNAC = NENG + 1. / 2 * (NENG - 2 * np.floor(NENG / 2.))
    THRUST = prop.sealevel_static_thrust * 1 / Units.lbf
    WTHR = 0.034 * THRUST * TNAC
    return WTHR * Units.lbs

## @ingroup Methods-Weights-Correlations-FLOPS
def misc_engine_FLOPS(vehicle,prop,nacelle):
    """ Calculates the miscellaneous engine weight based on the FLOPS method, electrical control system weight
        and starter engine weight
        
        Assumptions:
            1) All nacelles are identical
            2) The number of nacelles is the same as the number of engines 

        Source:
            The Flight Optimization System Weight Estimation Method

        Inputs:
            vehicle - data dictionary with vehicle properties                            [dimensionless]
                 -.design_mach_number: design mach number
            prop    - data dictionary for the specific network that is being estimated [dimensionless]
                -.number_of_engines: number of engines
                -.sealevel_static_thrust: sealevel static thrust of engine               [N]
            nacelle              
                -.diameter: diameter of nacelle                                          [m]
              
        Outputs:              
            WEC: electrical engine control system weight                                 [kg]
            WSTART: starter engine weight                                                [kg]

        Properties Used:
            N/A
    """
  
    NENG    = prop.number_of_engines
    THRUST  = prop.sealevel_static_thrust * 1 / Units.lbf
    WEC     = 0.26 * NENG * THRUST ** 0.5
    FNAC    = nacelle.diameter / Units.ft
    VMAX    = vehicle.design_mach_number
    WSTART  = 11.0 * NENG * VMAX ** 0.32 * FNAC ** 1.6
    return WEC * Units.lbs, WSTART * Units.lbs

## @ingroup Methods-Weights-Correlations-FLOPS
def fuel_system_FLOPS(vehicle, NENG):
    """ Calculates the weight of the fuel system based on the FLOPS method
        Assumptions:

        Source:
            The Flight Optimization System Weight Estimation Method

        Inputs:
            vehicle - data dictionary with vehicle properties                   [dimensionless]
                -.design_mach_number: design mach number
                -.mass_properties.max_zero_fuel: maximum zero fuel weight   [kg]

        Outputs:
            WFSYS: Fuel system weight                                       [kg]

        Properties Used:
            N/A
    """
    VMAX = vehicle.design_mach_number
    FMXTOT = vehicle.mass_properties.max_zero_fuel / Units.lbs
    WFSYS = 1.07 * FMXTOT ** 0.58 * NENG ** 0.43 * VMAX ** 0.34
    return WFSYS * Units.lbs

## @ingroup Methods-Weights-Correlations-FLOPS
def engine_FLOPS(vehicle, prop):
    """ Calculates the dry engine weight based on the FLOPS method
        Assumptions:
            Rated thrust per scaled engine and rated thurst for baseline are the same
            Engine weight scaling parameter is 1.15
            Enginge inlet weight scaling exponent is 1
            Baseline inlet weight is 0 lbs as in example files FLOPS
            Baseline nozzle weight is 0 lbs as in example files FLOPS

        Source:
            The Flight Optimization System Weight Estimation Method

        Inputs:
            vehicle - data dictionary with vehicle properties                   [dimensionless]
                -.systems.accessories: type of aircraft (short-range, commuter
                                                        medium-range, long-range,
                                                        sst, cargo)
            prop    - data dictionary for the specific network that is being estimated [dimensionless]
                -.sealevel_static_thrust: sealevel static thrust of engine  [N]

        Outputs:
            WENG: dry engine weight                                         [kg]

        Properties Used:
            N/A
    """
    EEXP = 1.15
    EINL = 1
    ENOZ = 1
    THRSO = prop.sealevel_static_thrust * 1 / Units.lbf
    THRUST = THRSO
    if vehicle.systems.accessories == "short-range" or vehicle.systems.accessories == "commuter":
        WENGB = THRSO / 10.5
    else:
        WENGB = THRSO / 5.5
    WINLB = 0 / Units.lbs
    WNOZB = 0 / Units.lbs
    WENGP = WENGB * (THRUST / THRSO) ** EEXP
    WINL = WINLB * (THRUST / THRSO) ** EINL
    WNOZ = WNOZB * (THRUST / THRSO) ** ENOZ
    WENG = WENGP + WINL + WNOZ
    return WENG * Units.lbs
