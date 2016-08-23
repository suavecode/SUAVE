# windmilling_drag_coefficient.py
#
# Created:  Jul 2014, T. Orra, C. Ilario, 
# Modified: Oct 2015, T. Orra
#           Jan 2016, E. Botero  

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE Imports
import SUAVE
from SUAVE.Components import Wings
from SUAVE.Core import Units, Data
from  SUAVE.Analyses import Results

# ----------------------------------------------------------------------
#  Compute drag of turbofan in windmilling condition
# ----------------------------------------------------------------------

def windmilling_drag(geometry,state):
    """ SUAVE.Methods.Aerodynamics.Fidelity_Zero.Drag.windmilling_drag(geometry,state):
        Computes the windmilling drag of turbofan engines

        Inputs:
            geometry   - data dictionary with data of vehicle and engine
            state      - to output drag breakdown

        Outputs:
            windmilling_drag

        Assumptions:
"""
    # ==============================================
	# Unpack
    # ==============================================
    vehicle = geometry

    # Defining reference area
    if vehicle.reference_area:
            reference_area = vehicle.reference_area
    else:
        n_wing = 0
        for wing in vehicle.wings:
            if not isinstance(wing,Wings.Main_Wing): continue
            n_wing = n_wing + 1
            reference_area = wing.sref
        if n_wing > 1:
            print ' More than one Main_Wing in the vehicle. Last one will be considered.'
        elif n_wing == 0:
            print  'No Main_Wing defined! Using the 1st wing found'
            for wing in vehicle.wings:
                if not isinstance(wing,Wings.Wing): continue
                reference_area = wing.sref
                break

    # getting geometric data from engine (estimating when not available)
    for idx,propulsor in enumerate(vehicle.propulsors):
        try:
            swet_nac = propulsor.areas.wetted
        except:
            try:
                D_nac = propulsor.nacelle_diameter
                if propulsor.engine_length <> 0.:
                    l_nac = propulsor.engine_length
                else:
                    try:
                        MMO = vehicle.max_mach_operational
                    except:
                        MMO = 0.84
                    D_nac_in = D_nac / Units.inches
                    l_nac = (2.36 * D_nac_in - 0.01*(D_nac_in*MMO)**2) * Units.inches
            except AttributeError:
                print 'Error calculating windmilling drag. Engine dimensions missing.'
            swet_nac = 5.62 * D_nac * l_nac

    # Compute
    windmilling_drag_coefficient = 0.007274 * swet_nac / reference_area

    # dump data to state
    windmilling_result = Results(
        wetted_area                  = swet_nac    ,
        windmilling_drag_coefficient = windmilling_drag_coefficient ,
    )
    state.conditions.aerodynamics.drag_breakdown.windmilling_drag = windmilling_result

    return windmilling_drag_coefficient