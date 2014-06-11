# payload.py
# 
# Created:  Andrew Wendorff, Jan 2014
# Modified: Andrew Wendorff, Feb 2014        


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Attributes import Units as Units
from SUAVE.Structure import (
    Data, Container, Data_Exception, Data_Warning,
)


# ----------------------------------------------------------------------
#   Method
# ----------------------------------------------------------------------

def payload(TOW, empty, num_pax, wt_cargo, wt_passenger = 195.,wt_baggage = 30.):
    """ output = SUAVE.Methods.Weights.Correlations.Tube_Wing.payload(TOW, empty, num_pax, wt_cargo)
        Calculate the weight of the payload and the resulting fuel mass
    
        Inputs:
            TOW -  [kilograms]
            wt_empty - Operating empty weight of the aircraft [kilograms]
            num_pax - number of passengers on the aircraft [dimensionless]
            wt_cargo - weight of cargo being carried on the aircraft [kilogram]
            wt_passenger - weight of each passenger on the aircraft [dimensionless]
            wt_baggage - weight of the baggage for each passenger [dimensionless]
        
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
    wt_pax     = wt_passenger * num_pax * Units.lb
    wt_bag     = wt_baggage * num_pax *Units.lb
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
