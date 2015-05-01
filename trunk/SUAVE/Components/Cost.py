
from SUAVE.Core import Data, Container, Data_Exception, Data_Warning
from Component import Component

class Cost(Component):
    def __defaults__(self):
        self.tag = 'Cost'
        self.depreciate_years = 0.0 
        self.fuel_price       = 0.0 
        self.oil_price        = 0.0 
        self.insure_rate      = 0.0 
        self.maintenance_rate = 0.0
        self.pilot_rate       = 0.0
        self.crew_rate        = 0.0
        self.inflator         = 0.0
        self.reference_dollars= 0.0
    
    