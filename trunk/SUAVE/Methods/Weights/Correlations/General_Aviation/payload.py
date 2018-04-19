<<<<<<< HEAD
# payload.py
# 
# Created:  Jan 2014, A. Wendorff
# Modified: Feb 2014, A. Wendorff
#           Feb 2016, E. Botero
=======
## @ingroup Methods-Weights-Correlations-General_Aviation
# payload.py
# 
# Created:  Feb 2018, M. Vegh
# Modified:
>>>>>>> develop

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units, Data

# ----------------------------------------------------------------------
#   Payload
# ----------------------------------------------------------------------
<<<<<<< HEAD

def payload(TOW, empty, num_pax, wt_cargo, wt_passenger = 225.*Units.lbs,wt_baggage = 0.):
    """ output = SUAVE.Methods.Weights.Correlations.Tube_Wing.payload(TOW, empty, num_pax, wt_cargo)
        Calculate the weight of the payload and the resulting fuel mass
    
        Inputs:
            TOW -  [kilograms]
            wt_empty - Operating empty weight of the aircraft [kilograms]
            num_pax - number of passengers on the aircraft [dimensionless]
            wt_cargo - weight of cargo being carried on the aircraft [kilogram]
            wt_passenger - weight of each passenger on the aircraft [kilogram]
            wt_baggage - weight of the baggage for each passenger [kilogram]
=======
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
>>>>>>> develop
            
            
        Outputs:
            output - a data dictionary with fields:
                payload - weight of the passengers plus baggage and paid cargo [kilograms]
<<<<<<< HEAD
                pax - weight of all the passengers [kilogram]
                bag - weight of all the baggage [kilogram]
                fuel - weight of the fuel carried[kilogram]
                empty - operating empty weight of the aircraft [kilograms]
            
        Assumptions:
            based on the total payload of a Cessna 172 Skyhawk at a full fuel load
            link: https://disciplesofflight.com/cessna-172-skyhawk/
    """     
    
=======
                pax - weight of all the passengers                             [kilograms]
                bag - weight of all the baggage                                [kilograms]
                fuel - weight of the fuel carried                              [kilograms]
                empty - operating empty weight of the aircraft                 [kilograms]

    """     

>>>>>>> develop
    # process
    wt_pax     = wt_passenger * num_pax 
    wt_bag     = wt_baggage * num_pax
    wt_payload = wt_pax + wt_bag + wt_cargo
<<<<<<< HEAD
    wt_fuel    = TOW - wt_payload - empty
    
=======

>>>>>>> develop
    # packup outputs
    output = Data()
    output.payload = wt_payload
    output.pax     = wt_pax   
    output.bag     = wt_bag
<<<<<<< HEAD
    output.fuel    = wt_fuel
    output.empty   = empty
  
=======
    output.empty   = empty

>>>>>>> develop
    return output