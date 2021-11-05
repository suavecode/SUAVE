## @defgroup Vehicle
# Vehicle.py
#
# Created:  ### 2013, SUAVE Team
# Modified: ### ####, M. Vegh
#           Feb 2016, E. Botero
#           Apr 2017, M. Clarke 
#           Apr 2020, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data, DataOrdered
from SUAVE import Components
from SUAVE.Components import Physical_Component
import numpy as np

from warnings import warn
import string
chars = string.punctuation + string.whitespace
t_table = str.maketrans( chars          + string.ascii_uppercase , 
                            '_'*len(chars) + string.ascii_lowercase )

# ----------------------------------------------------------------------
#  Vehicle Data Class
# ----------------------------------------------------------------------

## @ingroup Vehicle
class Vehicle(Data):
    """SUAVE Vehicle container class with database + input / output functionality
    
    Assumptions:
    None
    
    Source:
    None
    """    

    def __defaults__(self):
        """This sets the default values.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
            """          
        self.tag = 'vehicle'
        self.fuselages              = Components.Fuselages.Fuselage.Container()
        self.wings                  = Components.Wings.Wing.Container()
        self.networks               = Components.Energy.Networks.Network.Container()
        self.nacelles               = Components.Nacelles.Nacelle.Container()
        self.systems                = Components.Systems.System.Container()
        self.mass_properties        = Vehicle_Mass_Container()
        self.payload                = Components.Payloads.Payload.Container()
        self.costs                  = Costs()
        self.envelope               = Components.Envelope()
        self.landing_gear           = Components.Landing_Gear.Landing_Gear.Container()
        self.reference_area         = 0.0
        self.passengers             = 0.0
        self.performance            = DataOrdered()

    _component_root_map = None

    def __init__(self,*args,**kwarg):
        """ Sets up the component hierarchy for a vehicle
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
        """          
        # will set defaults
        super(Vehicle,self).__init__(*args,**kwarg)

        self._component_root_map = {
            Components.Fuselages.Fuselage              : self['fuselages']        ,
            Components.Wings.Wing                      : self['wings']            ,
            Components.Systems.System                  : self['systems']          ,
            Components.Energy.Networks.Network         : self['networks']         ,
            Components.Nacelles.Nacelle                : self['nacelles']         ,
            Components.Envelope                        : self['envelope']         ,
            Components.Landing_Gear.Landing_Gear       : self['landing_gear']     ,
            Vehicle_Mass_Properties                    : self['mass_properties']  ,
        }
        
        self.append_component(Vehicle_Mass_Properties())
        
        return

    def find_component_root(self,component):
        """ find pointer to component data root.
        
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
        """  

        component_type = type(component)

        # find component root by type, allow subclasses
        for component_type, component_root in self._component_root_map.items():
            if isinstance(component,component_type):
                break
        else:
            raise Exception("Unable to place component type %s" % component.typestring())

        return component_root


    def append_component(self,component):
        """ adds a component to vehicle
            
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
        """  

        # assert database type
        if not isinstance(component,Data):
            raise Exception('input component must be of type Data()')

        # find the place to store data
        component_root = self.find_component_root(component)
        
        # See if the component exists, if it does modify the name
        keys = component_root.keys()
        if str.lower(component.tag) in keys:
            string_of_keys = "".join(component_root.keys())
            n_comps = string_of_keys.count(component.tag)
            component.tag = component.tag + str(n_comps+1)

        # store data
        component_root.append(component)

        return

    def sum_mass(self):
        """ Regresses through the vehicle and sums the masses
        
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
        """  

        total = 0.0
        
        for key in self.keys():
            item = self[key]
            if isinstance(item,Physical_Component.Container):
                total += item.sum_mass()

        return total
    
    
    def center_of_gravity(self):
        """ will recursively search the data tree and sum
            any Comp.Mass_Properties.mass, and return the total sum
            
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
        """   
        total = np.array([[0.0,0.0,0.0]])

        for key in self.keys():
            item = self[key]
            if isinstance(item,Physical_Component.Container):
                total += item.total_moment()
                
        mass = self.sum_mass()
        if mass ==0:
            mass = 1.
                
        CG = total/mass
        
        self.mass_properties.center_of_gravity = CG
                
        return CG


## @ingroup Vehicle
class Vehicle_Mass_Properties(Components.Mass_Properties):

    """ Vehicle_Mass_Properties():
        The vehicle's mass properties.

    
    Assumptions:
    None
    
    Source:
    None
    """

    def __defaults__(self):
        """This sets the default values.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
            """         

        self.tag             = 'mass_properties'
        self.operating_empty = 0.0
        self.max_takeoff     = 0.0
        self.takeoff         = 0.0
        self.max_landing     = 0.0
        self.landing         = 0.0
        self.max_cargo       = 0.0
        self.cargo           = 0.0
        self.max_payload     = 0.0
        self.payload         = 0.0
        self.passenger       = 0.0
        self.crew            = 0.0
        self.max_fuel        = 0.0
        self.fuel            = 0.0
        self.max_zero_fuel   = 0.0
        self.center_of_gravity = [[0.0,0.0,0.0]]
        self.zero_fuel_center_of_gravity = np.array([[0.0,0.0,0.0]])

        self.generative_design_max_per_vehicle = 1
        self.generative_design_special_parent  = None
        self.generative_design_characteristics = ['max_takeoff','max_zero_fuel']
        self.generative_design_minimum         = 1
        self.generative_design_char_min_bounds = [1,1]   
        self.generative_design_char_max_bounds = [np.inf,np.inf]        

## @ingroup Vehicle
class Costs(Data):
    """ Costs class for organizing the costs of things

    Assumptions:
    None
    
    Source:
    None
    """    
    def __defaults__(self):
        """This sets the default values.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
            """         
        self.tag = 'costs'
        self.industrial = Components.Costs.Industrial_Costs()
        self.operating  = Components.Costs.Operating_Costs()
        
        
class Vehicle_Mass_Container(Components.Physical_Component.Container,Vehicle_Mass_Properties):
        
    def append(self,value,key=None):
        """ Appends the vehicle mass, but only let's one ever exist. Keeps the newest one
        
        Assumptions:
        None
    
        Source:
        N/A
    
        Inputs:
        None
    
        Outputs:
        None
    
        Properties Used:
        N/A
        """      
        self.clear()
        for key in value.keys():
            self[key] = value[key]

    def get_children(self):
        """ Returns the components that can go inside
        
        Assumptions:
        None
    
        Source:
        N/A
    
        Inputs:
        None
    
        Outputs:
        None
    
        Properties Used:
        N/A
        """       
        
        return [Vehicle_Mass_Properties]
