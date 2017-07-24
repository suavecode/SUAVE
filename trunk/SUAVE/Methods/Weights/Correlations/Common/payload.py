# payload.py
# 
# Created:  Jan 2014, A. Wendorff
# Modified: Feb 2014, A. Wendorff
#           Feb 2016, E. Botero
#           Jul 2017, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units, Data

# ----------------------------------------------------------------------
#   Payload
# ----------------------------------------------------------------------

def payload(TOW, empty, num_pax, wt_cargo, wt_passenger = 195*Units.lbs,wt_baggage = 30*Units.lbs):
    """ output = SUAVE.Methods.Weights.Correlations.Common.payload(TOW, empty, num_pax, wt_cargo)
        Calculate the weight of the payload and the resulting fuel mass
    
        Inputs:
            TOW -  [kilograms]
            wt_empty - Operating empty weight of the aircraft [kilograms]
            num_pax - number of passengers on the aircraft [dimensionless]
            wt_cargo - weight of cargo being carried on the aircraft [kilogram]
            wt_passenger - weight of each passenger on the aircraft [kilogram]
            wt_baggage - weight of the baggage for each passenger [kilogram]
        
        Outputs:
            output - a data dictionary with fields:
                payload - weight of the passengers plus baggage and paid cargo [kilograms]
                pax - weight of all the passengers [kilogram]
                bag - weight of all the baggage [kilogram]
                fuel - weight of the fuel carried[kilogram]
                empty - operating empty weight of the aircraft [kilograms]
            
        Assumptions:
            based on FAA guidelines for weight of passengers 
    """
    
    # process
    wt_pax     = wt_passenger * num_pax
    wt_bag     = wt_baggage * num_pax 
    wt_payload = wt_pax + wt_bag + wt_cargo
    wt_fuel    = TOW - wt_payload - empty
    
    # packup outputs
    output = Data()
    output.payload = wt_payload
    output.pax     = wt_pax   
    output.bag     = wt_bag
    output.fuel    = wt_fuel
    output.empty   = empty
  
    return output