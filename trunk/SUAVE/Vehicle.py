## @defgroup Vehicle
# Vehicle.py
#
# Created:  ### 2013, SUAVE Team
# Modified: ### ####, M. Vegh
#           Feb 2016, E. Botero
#           Apr 2017, M. Clarke 
#           Apr 2020, E. Botero
#           Jan 2022, S. Karpuk

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data, DataOrdered, Units
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
        self.constraints            = Constraints()
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

class Constraints(Data):
    """ Constraints class for the constraint analysis

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
            
        self.tag                = 'Constraint analysis'
        self.plot_tag           = False


        # Defines default constraint analyses
        self.analyses            = Data()
        self.analyses.takeoff    = Data()
        self.analyses.cruise     = Data()
        self.analyses.max_cruise = Data()
        self.analyses.landing    = Data()
        self.analyses.OEI_climb  = Data()
        self.analyses.turn       = Data()
        self.analyses.climb      = Data()
        self.analyses.ceiling    = Data()

        self.analyses.takeoff.compute    = True
        self.analyses.cruise.compute     = True
        self.analyses.max_cruise.compute = False
        self.analyses.landing.compute    = True
        self.analyses.OEI_climb.compute  = True
        self.analyses.turn.compute       = False
        self.analyses.climb.compute      = False
        self.analyses.ceiling.compute    = False
        
        self.wing_loading      = np.arange(2, 200, 0.25) * Units['force_pound/foot**2']
        self.design_point_type = None

        # Default parameters for the constraint analysis
        # take-off
        self.analyses.takeoff.runway_elevation     = 0.0
        self.analyses.takeoff.ground_run           = 0.0
        self.analyses.takeoff.rolling_resistance   = 0.05
        self.analyses.takeoff.liftoff_speed_factor = 1.1
        self.analyses.takeoff.delta_ISA            = 0.0
        # climb
        self.analyses.climb.altitude   = 0.0
        self.analyses.climb.airspeed   = 0.0
        self.analyses.climb.climb_rate = 0.0
        self.analyses.climb.delta_ISA  = 0.0
        # OEI climb
        self.analyses.OEI_climb.climb_speed_factor = 1.2
        # cruise
        self.analyses.cruise.altitude        = 0.0
        self.analyses.cruise.delta_ISA       = 0.0
        self.analyses.cruise.airspeed        = 0.0
        self.analyses.cruise.thrust_fraction = 0.0
        # max cruise
        self.analyses.max_cruise.altitude        = 0.0
        self.analyses.max_cruise.delta_ISA       = 0.0
        self.analyses.max_cruise.mach            = 0.0
        self.analyses.max_cruise.thrust_fraction = 0.0
        # turn
        self.analyses.turn.angle           = 0.0
        self.analyses.turn.altitude        = 0.0
        self.analyses.turn.delta_ISA       = 0.0
        self.analyses.turn.mach            = 0.0
        self.analyses.turn.specific_energy = 0.0
        self.analyses.turn.thrust_fraction = 0.0
        # ceiling
        self.analyses.ceiling.altitude  = 0.0
        self.analyses.ceiling.delta_ISA = 0.0
        self.analyses.ceiling.mach      = 0.0
        # landing
        self.analyses.landing.ground_roll           = 0.0
        self.analyses.landing.approach_speed_factor = 1.23
        self.analyses.landing.runway_elevation      = 0.0
        self.analyses.landing.delta_ISA             = 0.0 

        # Default aircraft properties
        # geometry
        self.geometry = Data()
        self.geometry.aspect_ratio                  = 0.0 
        self.geometry.taper                         = 0.0
        self.geometry.thickness_to_chord            = 0.0
        self.geometry.sweep_quarter_chord           = 0.0
        self.geometry.high_lift_configuration_type  = None
        # engine
        self.engine = Data()
        self.engine.type                    = None
        self.engine.number                  = 0
        self.engine.bypass_ratio            = 0.0 
        self.engine.throttle_ratio          = 1.0   
        self.engine.afterburner             = False
        self.engine.method                  = 'Mattingly'
        # propeller
        self.propeller = Data()
        self.propeller.takeoff_efficiency   = 0.0
        self.propeller.climb_efficiency     = 0.0
        self.propeller.cruise_efficiency    = 0.0
        self.propeller.turn_efficiency      = 0.0
        self.propeller.ceiling_efficiency   = 0.0
        self.propeller.OEI_climb_efficiency = 0.0

        # Define aerodynamics
        self.aerodynamics = Data()
        self.aerodynamics.oswald_factor   = 0.0
        self.aerodynamics.cd_takeoff      = 0.0   
        self.aerodynamics.cl_takeoff      = 0.0   
        self.aerodynamics.cl_max_takeoff  = 0.0
        self.aerodynamics.cl_max_landing  = 0.0  
        self.aerodynamics.cd_min_clean    = 0.0
        self.aerodynamics.fuselage_factor = 0.974
        self.aerodynamics.viscous_factor  = 0.38

        