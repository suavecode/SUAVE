# test_aerodynamics.py

#import sys
#sys.path.append('../trunk')


import SUAVE
import numpy
from SUAVE.Attributes.Missions.Segments import Base_Segment
from SUAVE import Vehicle
from SUAVE.Components.Wings import Wing
from SUAVE.Components.Fuselages import Fuselage
from SUAVE.Components.Propulsors import Turbofan
from SUAVE.Geometry.Two_Dimensional.Planform import wing_planform
from SUAVE.Geometry.Two_Dimensional.Planform import fuselage_planform
from SUAVE.Structure import Data

from SUAVE.Attributes.Aerodynamics import PASS_Aero

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# MAIN

# MAIN
def main():
    
    test()
    
    return
    

# TEST AERODYNAMICS
def test():
    
    aircraft=Vehicle()
    aircraft.tag="aero_test"
    aircraft.S=125.00
    
    Wing1=Wing(tag = 'Wing1')
    
    Wing1.sref      = 125.00 #124.862
    Wing1.ar        = 9
    Wing1.span      = 35.66
    Wing1.sweep     = 20.0*numpy.pi/180
    Wing1.symmetric = True
    Wing1.t_c       = 0.1
    Wing1.taper     = 0.16
    
    wing_planform(Wing1)
    
    Wing1.chord_mac   = 12.5
    Wing1.S_exposed   = 0.8*Wing1.area_wetted
    Wing1.S_affected  = 0.6*Wing1.area_wetted 
    #Wing1.Cl         = 0.3
    Wing1.e           = 0.9
    Wing1.twist_rc    = 3.0*numpy.pi/180
    Wing1.twist_tc    = -1.0*numpy.pi/180
 
    aircraft.append_component(Wing1)
  
  
    Wing2=Wing(tag = 'Wing2')
        
    Wing2.sref      = 32.488
    Wing2.ar        =  6.16
    Wing2.span      = 14.146
    Wing2.sweep     = 35.0*numpy.pi/180
    Wing2.symmetric = True
    Wing2.t_c       = 0.08
    Wing2.taper     = 0.4
    
    wing_planform(Wing2)
    
    #Wing2.chord_mac = 12.5
    Wing2.chord_mac  = 8.0
    Wing2.S_exposed  = 0.8*Wing2.area_wetted
    Wing2.S_affected = 0.6*Wing2.area_wetted     
    #Wing2.Cl        = 0.2
    Wing2.e          = 0.9
    Wing2.twist_rc   = 3.0*numpy.pi/180
    Wing2.twist_tc   = 0.0*numpy.pi/180   
  
    aircraft.append_component(Wing2)
    
    
    Wing3=Wing(tag = 'Wing3')
        
    Wing3.sref      = 32.488
    Wing3.ar        = 1.91
    Wing3.span      = 7.877
    Wing3.sweep     = 0.0*numpy.pi/180
    Wing3.symmetric = False
    Wing3.t_c       = 0.08
    Wing3.taper     = 0.25
       
    wing_planform(Wing3)
    
    Wing3.chord_mac  = 8.0
    Wing3.S_exposed  = 0.8*Wing3.area_wetted
    Wing3.S_affected = 0.6*Wing3.area_wetted     
    #Wing3.Cl        = 0.002  
    Wing3.e          = 0.9
    Wing3.twist_rc   = 0.0*numpy.pi/180
    Wing3.twist_tc   = 0.0*numpy.pi/180   
    Wing3.vertical   = True
        
    aircraft.append_component(Wing3)
   
   
    fus=Fuselage(tag = 'fuselage1')
    
    fus.num_coach_seats = 200
    fus.seat_pitch      = 1
    fus.seats_abreast   = 6
    fus.fineness_nose   = 1.6
    fus.fineness_tail   =  2
    fus.fwdspace        = 6
    fus.aftspace        = 5
    fus.width           = 4
    fus.height          = 4   
    
    fuselage_planform(fus)
    
    aircraft.append_component(fus)

    turbofan=Turbofan()
    turbofan.nacelle_dia= 4.0
    aircraft.append_component(turbofan)

    wing_aero = SUAVE.Attributes.Aerodynamics.Fidelity_Zero()
    wing_aero.initialize(aircraft)
    aircraft.Aerodynamics = wing_aero 


    Seg=Base_Segment()
    Seg.p   = 23900.0            # Pa
    Seg.T   = 218.0            # K
    Seg.rho = 0.41          # kg/m^3
    Seg.mew = 1.8*10**-5          # Ps-s
    Seg.M   = 0.8            # dimensionless


    
    conditions = Data()
    conditions.freestream   = Data()
    conditions.aerodynamics = Data()

    # freestream conditions
    #conditions.freestream.velocity           = ones_1col * 0
    conditions.freestream.mach_number        = Seg.M
    conditions.freestream.pressure           = Seg.p   
    conditions.freestream.temperature        = Seg.T  
    conditions.freestream.density            = Seg.rho
    #conditions.freestream.speed_of_sound     = ones_1col * 0
    conditions.freestream.viscosity          = Seg.mew
    #conditions.freestream.altitude           = ones_1col * 0
    #conditions.freestream.gravity            = ones_1col * 0
    #conditions.freestream.reynolds_number    = ones_1col * 0
    #conditions.freestream.dynamic_pressure   = ones_1col * 0
    
    # aerodynamics conditions
    conditions.aerodynamics.angle_of_attack  = 0.  
    conditions.aerodynamics.side_slip_angle  = 0.
    conditions.aerodynamics.roll_angle       = 0.
    conditions.aerodynamics.lift_coefficient = 0.
    conditions.aerodynamics.drag_coefficient = 0.
    conditions.aerodynamics.lift_breakdown   = Data()
    conditions.aerodynamics.drag_breakdown   = Data()

    
    [Cl,Cd]=aircraft.Aerodynamics(conditions)
  
    print 'Aerodynamics module test script'
    print 'aircraft Cl' , Cl
    print 'aircraft Cd' , Cd
  
    return


# call main
if __name__ == '__main__':
    main()
