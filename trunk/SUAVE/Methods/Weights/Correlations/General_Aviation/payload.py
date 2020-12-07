## @ingroup Methods-Weights-Correlations-General_Aviation
# payload.py
# 
# Created:  Feb 2018, M. Vegh
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units, Data

# ----------------------------------------------------------------------
#   Payload
# ----------------------------------------------------------------------
## @ingroup Methods-Weights-Correlations-General_Aviation
def payload(TOW, empty, num_pax, wt_cargo, wt_passenger = 225.*Units.lbs,wt_baggage = 0.):
    """ 
        Calculate the weight of the payload and the resulting fuel mass
 
        Inputs:
            TOW -                                                    [kilograms]
            wt_empty - Operating empty weight of the aircraft        [kilograms]
            num_pax - number of passengers on the aircraft           [dimensionless]
            wt_cargo - weight of cargo being carried on the aircraft [kilogram]
            wt_passenger - weight of each passenger on the aircraft  [kilograms]
            wt_baggage - weight of the baggage for each passenger    [kilograms]
            
            
        Outputs:
            output - a data dictionary with fields:
                payload - weight of the passengers plus baggage and paid cargo [kilograms]
                pax - weight of all the passengers                             [kilograms]
                bag - weight of all the baggage                                [kilograms]
                fuel - weight of the fuel carried                              [kilograms]
                empty - operating empty weight of the aircraft                 [kilograms]

    """     

    # process
    wt_pax     = wt_passenger * num_pax 
    wt_bag     = wt_baggage * num_pax
    wt_payload = wt_pax + wt_bag + wt_cargo

    # packup outputs
    output = Data()
    output.total        = wt_payload
    output.passengers   = wt_pax
    output.baggage      = wt_bag
    output.cargo        = wt_cargo

    return output