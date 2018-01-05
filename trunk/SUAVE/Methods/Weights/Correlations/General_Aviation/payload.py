## @ingroup Methods-Weights-Correlations-General_Aviation
# payload.py
# 
# Created:  Jan 2014, A. Wendorff
# Modified: Feb 2014, A. Wendorff
#           Feb 2016, E. Botero

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
            
        Source:
            based on the total payload of a Cessna 172 Skyhawk at a full fuel load
            link: https://disciplesofflight.com/cessna-172-skyhawk/
    """     
    
    # process
    wt_pax     = wt_passenger * num_pax 
    wt_bag     = wt_baggage * num_pax
    wt_payload = wt_pax + wt_bag + wt_cargo
    #wt_fuel    = TOW - wt_payload - empty
    
    # packup outputs
    output = Data()
    output.payload = wt_payload
    output.pax     = wt_pax   
    output.bag     = wt_bag
    #output.fuel    = wt_fuel
    
    output.empty   = empty
  
    return output