# Vehicle.py
# 
# Created:  ### 2013, SUAVE Team
# Modified: ### ####, M. Vegh
#           Feb 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data, Container
from SUAVE import Components
import numpy as np

# ----------------------------------------------------------------------
#  Vehicle Data Class
# ----------------------------------------------------------------------

class Vehicle(Data):
    ''' SUAVE.Vehicle(**kwarg)
        Arbitrary Vehicle Initialization
        Vehicle.py: SUAVE Vehicle container class with database + input / output functionality

        Inputs:
            optional, dictionary of data for initialization

    '''

    def __defaults__(self):
        self.tag = 'vehicle'
        self.fuselages       = Components.Fuselages.Fuselage.Container()
        self.wings           = Components.Wings.Wing.Container()
        self.propulsors      = Components.Propulsors.Propulsor.Container()
        self.energy          = Components.Energy.Energy()
        self.systems         = Components.Systems.System.Container()
        self.mass_properties = Vehicle_Mass_Properties()
        self.cost            = Components.Cost()
        self.envelope        = Components.Envelope()
        self.reference_area  = 0.0
        self.passengers      = 0.0

        self.max_lift_coefficient_factor = 1.0

    _component_root_map = None

    def __init__(self,*args,**kwarg):
        # will set defaults
        super(Vehicle,self).__init__(*args,**kwarg)

        self._component_root_map = {
            Components.Fuselages.Fuselage              : self['fuselages']              ,
            Components.Wings.Wing                      : self['wings']                  ,
            Components.Systems.System                  : self['systems']                ,
            Components.Cost                            : self['cost']                   ,
            Components.Propulsors.Propulsor            : self['propulsors']             ,
            Components.Envelope                        : self['envelope']               ,
        }

        return

    def find_component_root(self,component):
        """ find pointer to component data root.
        """

        component_type = type(component)

        # find component root by type, allow subclasses
        for component_type, component_root in self._component_root_map.iteritems():
            if isinstance(component,component_type):
                break
        else:
            raise Exception , "Unable to place component type %s" % component.typestring()

        return component_root


    def append_component(self,component):
        """ adds a component to vehicle """

        # assert database type
        if not isinstance(component,Data):
            raise Exception, 'input component must be of type Data()'

        # find the place to store data
        component_root = self.find_component_root(component)

        # store data
        component_root.append(component)

        return


class Vehicle_Mass_Properties(Components.Mass_Properties):

    """ Vehicle_Mass_Properties():
        The vehicle's mass properties.  includes high level weight statement values

        Attributes:

            max_takeoff
            max_zero

            takeoff
            cargo
            operating_empty
            payload
            passenger
            crew
            fuel

            center_of_gravity
            Moments_Of_Inertia :
                center
                tensor

    """

    def __defaults__(self):

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
        self.zero_fuel_center_of_gravity=np.array([0.0,0.0,0.0])