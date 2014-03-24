# test_Weights.py

import SUAVE
import numpy as np
from SUAVE.Attributes import Units as Units
from SUAVE.Methods.Weights.Correlations import Propulsion as Propulsion
from SUAVE.Methods.Weights.Correlations import Tube_Wing as Tube_Wing
from SUAVE.Structure import (
    Data, Container, Data_Exception, Data_Warning,
)

# new units style
a = 4 * Units.mm # convert into base units
b = a / Units.mm # convert out of base units

engine = Data()
wing = Data()
aircraft = Data()
fuselage = Data()
horizontal = Data()
vertical = Data()

# Parameters Required
aircraft.Nult      = 3.5                       # Ultimate load
aircraft.TOW       = 200000. * Units.kilograms # Maximum takeoff weight in kilograms
aircraft.zfw       = 150000. * Units.kilograms # Maximum zero fuel weight in kilograms
aircraft.Nlim      = 1.5                       # Limit Load
aircraft.num_eng   = 2.                        # Number of engines on the aircraft
aircraft.num_pax   = 125.                      # Number of passengers
aircraft.wt_cargo  = 10000.  * Units.kilogram  # Mass of cargo
aircraft.num_seats = 125.                      # Number of seats on aircraft
aircraft.ctrl      = "fully powered"           # Specify fully powered, partially powered or anything else is fully aerodynamic
aircraft.ac        = "long-range"              # Specify what type of aircraft you have
aircraft.w2h       = 20.     * Units.meters    # Length from the mean aerodynamic center of wing to mean aerodynamic center of the horizontal tail

wing.gross_area    = 500.    * Units.meter**2  # Wing gross area in square meters
wing.span          = 50.     * Units.meter     # Span in meters
wing.taper         = 0.2                       # Taper ratio
wing.t_c           = 0.08                      # Thickness-to-chord ratio
wing.sweep         = 35.     * Units.deg       # sweep angle in degrees
wing.c_r           = 15.     * Units.meter     # Wing root chord length
wing.mac           = 10.     * Units.meters    # Length of the mean aerodynamic chord of the wing

fuselage.area      = 10.     * Units.meter**2  # Fuselage cross-sectional area 
fuselage.diff_p    = 10**5   * Units.pascal    # Maximum differential pressure
fuselage.width     = 5.      * Units.meter     # Width of the fuselage
fuselage.height    = 4.5     * Units.meter     # Height of the fuselage
fuselage.length    = 60.     * Units.meter     # Length of the fuselage

engine.thrust_sls  = 1000.   * Units.newton    # Define Thrust in Newtons

horizontal.area    = 75.     * Units.meters**2 # Area of the horizontal tail
horizontal.span    = 15.     * Units.meters    # Span of the horizontal tail
horizontal.sweep   = 38.     * Units.deg       # Sweep of the horizontal tail
horizontal.mac     = 5.      * Units.meters    # Length of the mean aerodynamic chord of the horizontal tail
horizontal.t_c     = 0.07                      # Thickness-to-chord ratio of the horizontal tail
horizontal.exposed = 1                         # Fraction of horizontal tail area exposed

vertical.area      = 60.     * Units.meters**2 # Area of the vertical tail
vertical.span      = 15.     * Units.meters    # Span of the vertical tail
vertical.t_c       = 0.07                      # Thickness-to-chord ratio of the vertical tail
vertical.sweep     = 40.     * Units.deg       # Sweep of the vertical tail
vertical.t_tail    = "no"                      # Set to "yes" for a T-tail

weight = Tube_Wing.empty(engine,wing,aircraft,fuselage,horizontal,vertical)

print(weight.empty)