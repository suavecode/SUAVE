# test_.py
# 
# Created:  Emilio Botero, Jun 2014
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Attributes import Units as Units
import numpy as np
from SUAVE.Structure import (
    Data, Container, Data_Exception, Data_Warning,
)

import empty as empty

wing       = Data()
horizontal = Data()
vertical   = Data()
aircraft   = Data()

wing.sref   = 31.00
wing.span   = 34.10
wing.mac    = wing.sref/wing.span
wing.Nwr    = 2*41+22
wing.deltaw = (wing.span**2)/(wing.Nwr*wing.sref)
wing.t_c    = 0.128
wing.Nwer   = 10.

horizontal.area   = 10. #FAKE NUMBER!!!!!
horizontal.span   = wing.sref*((1.+3./16.)/9.)
horizontal.mac    = horizontal.area/horizontal.span
horizontal.Nwr    = 16.
horizontal.deltah = (horizontal.span**2)/(horizontal.Nwr*horizontal.area)
horizontal.t_c    = 0.12 # I have no idea

vertical.area   = 10. #FAKE NUMBER!!!!!
vertical.span   = wing.sref*((11./16.)/9.)
vertical.mac    = horizontal.area/horizontal.span
vertical.Nwr    = 10.
vertical.deltah = (vertical.span**2)/(vertical.Nwr*vertical.area)
vertical.t_c    = 0.12 # I have no idea

aircraft.nult = 1.75
aircraft.gw   = 109633.3/1000.
aircraft.qm   = 28.87
aircraft.Ltb  = wing.sref*((2.+1./8.)/9.)


weights = empty.empty(wing,aircraft,horizontal,vertical)

#print weights