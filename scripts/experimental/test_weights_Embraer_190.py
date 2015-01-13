# test_Weights.py

import SUAVE
import numpy as np
from SUAVE.Attributes import Units as Units
from SUAVE.Methods.Weights.Correlations import Propulsion as Propulsion
from SUAVE.Methods.Weights.Correlations import Tube_Wing as Tube_Wing
from SUAVE.Core import (
    Data, Container, Data_Exception, Data_Warning,
)

def main():

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
    aircraft.Nult      = 1.5 * 2.5                       # Ultimate load
    aircraft.TOW       = 52300.  * Units.kilograms # Maximum takeoff weight in kilograms
    aircraft.zfw       = 42600. * Units.kilograms # Maximum zero fuel weight in kilograms
    aircraft.Nlim      = 2.5                       # Limit Load
    aircraft.num_eng   = 2.                        # Number of engines on the aircraft
    aircraft.num_pax   = 110.                      # Number of passengers
    aircraft.wt_cargo  = 0.  * Units.kilogram  # Mass of cargo
    aircraft.num_seats = 110.                      # Number of seats on aircraft
    aircraft.ctrl      = "partially powered"       # Specify fully powered, partially powered or anything else is fully aerodynamic
    aircraft.ac        = "medium-range"              # Specify what type of aircraft you have
    aircraft.w2h       = 16.     * Units.meters    # Length from the mean aerodynamic center of wing to mean aerodynamic center of the horizontal tail

    wing.gross_area    = 92.    * Units.meter**2  # Wing gross area in square meters
    wing.span          = 27.8     * Units.meter     # Span in meters
    wing.taper         = 0.28                       # Taper ratio
    wing.t_c           = 0.105                      # Thickness-to-chord ratio
    wing.sweep         = 23.5     * Units.deg       # sweep angle in degrees
    wing.c_r           = 5.4     * Units.meter     # Wing exposed root chord length
    wing.mac           = 12.     * Units.ft    # Length of the mean aerodynamic chord of the wing

    fuselage.area      = 320.      * Units.meter**2  # Fuselage wetted area 
    fuselage.diff_p    = 8.5     * Units.force_pound / Units.inches**2    # Maximum differential pressure
    fuselage.width     = 3.      * Units.meter     # Width of the fuselage
    fuselage.height    = 3.35    * Units.meter     # Height of the fuselage
    fuselage.length    = 36.24     * Units.meter     # Length of the fuselage

    engine.thrust_sls  = 18500.   * Units.force_pound    # Define Thrust in Newtons

    horizontal.area    = 26.     * Units.meters**2 # Area of the horizontal tail
    horizontal.span    = 12.08     * Units.meters    # Span of the horizontal tail
    horizontal.sweep   = 34.5     * Units.deg       # Sweep of the horizontal tail
    horizontal.mac     = 2.4      * Units.meters    # Length of the mean aerodynamic chord of the horizontal tail
    horizontal.t_c     = 0.11                      # Thickness-to-chord ratio of the horizontal tail
    horizontal.exposed = 0.9                         # Fraction of horizontal tail area exposed

    vertical.area      = 16.     * Units.meters**2 # Area of the vertical tail
    vertical.span      = 5.3     * Units.meters    # Span of the vertical tail
    vertical.t_c       = 0.12                      # Thickness-to-chord ratio of the vertical tail
    vertical.sweep     = 35.     * Units.deg       # Sweep of the vertical tail
    vertical.t_tail    = "no"                      # Set to "yes" for a T-tail

    aircraft.weight = Tube_Wing.empty(engine,wing,aircraft,fuselage,horizontal,vertical)

    outputWeight(aircraft,'weight_EMB190.dat')

def outputWeight(aircraft,filename):

    import time                     # importing library
    import datetime                 # importing library

    #unpack
    weight = aircraft.weight
    ac_type = aircraft.ac
    ctrl_type = aircraft.ctrl

    fid = open(filename,'w')   # Open output file
    fid.write('Output file with weight breakdown\n\n') #Start output printing
    fid.write('\n')
    fid.write( ' Airplane type: .................. : ' + ac_type.upper() + '\n')
    fid.write( ' Flight Controls type: ........... : ' + ctrl_type.upper() + '\n')
    fid.write('\n')
    fid.write( ' Wing weight ..................... : ' + str( '%8.0F'   %   weight.wing )  + ' kg\n' )
    fid.write( ' Fuselage weight ................. : ' + str( '%8.0F'   %   weight.fuselage )  + ' kg\n' )
    fid.write( ' Propulsion system weight ........ : ' + str( '%8.0F'   %   weight.propulsion )  + ' kg\n' )
    fid.write( ' Landing Gear weight ............. : ' + str( '%8.0F'   %   weight.landing_gear )  + ' kg\n' )
    fid.write( ' Horizontal Tail weight .......... : ' + str( '%8.0F'   %   weight.horizontal_tail )  + ' kg\n' )
    fid.write( ' Vertical Tail weight .............: ' + str( '%8.0F'   %   weight.vertical_tail  ) + ' kg\n' )
    fid.write( ' Rudder weight ....................: ' + str( '%8.0F'   %   weight.rudder  ) + ' kg\n' )
    fid.write( ' Systems weight ...................: ' + str( '%8.0F'   %   weight.systems  ) + ' kg\n' )
    fid.write( '\n'  )
    fid.write( ' Empty weight .....................: ' + str( '%8.0F'   %   weight.empty  ) + ' kg\n' )
    fid.write( ' .................................................\n'  )
    fid.write(2*'\n')

    # Print timestamp
    fid.write(2*'\n'+ 43*'-'+ '\n' + datetime.datetime.now().strftime(" %A, %d. %B %Y %I:%M:%S %p"))

# ----------------------------------------------------------------------
#   Call Main
# ----------------------------------------------------------------------
if __name__ == '__main__':
    main()