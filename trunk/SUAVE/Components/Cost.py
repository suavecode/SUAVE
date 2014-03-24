
from SUAVE.Structure import Data, Container, Data_Exception, Data_Warning
from Component import Component

class Cost(Component):
    def __defaults__(self):
        self.tag = 'Cost'
        self.depreciate_years = 0.0 
        self.fuel_price       = 0.0 
        self.oil_price        = 0.0 
        self.insure_rate      = 0.0 
        self.labor_rate       = 0.0 
        self.inflator         = 0.0
    
    