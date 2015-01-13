# test_Concorde.py
# 
# Created:  Tim MacDonald, 6/25/14
# Modified: Tim MacDonald, 7/10/14

""" evaluate a mission with a Concorde
"""


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Attributes import Units
from SUAVE.Core import \
    Data, Container, Data_Exception, Data_Warning


import numpy as np
import pylab as plt

import copy, time


# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():
    
    # build the vehicle
    vehicle = define_vehicle()
    
    # define the mission
    
    # evaluate the mission
    results = evaluate_polar(vehicle)
    
    # plot results
    post_process(vehicle,results)
    
    return


# ----------------------------------------------------------------------
#   Build the Vehicle
# ----------------------------------------------------------------------

def define_vehicle():
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Concorde'

    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    
    # Data mostly from www.concordesst.com

    # mass properties
    vehicle.Mass_Props.m_full       = 186880   # kg
    vehicle.Mass_Props.m_empty      = 78700    # kg
    vehicle.Mass_Props.m_takeoff    = 185000   # kg
    vehicle.Mass_Props.m_flight_min = 100000   # kg - Note: Actual value is unknown

    # basic parameters
    vehicle.delta    = 55.0                     # deg
    vehicle.S        = 358.25                   # m^2
    vehicle.A_engine = np.pi*(1.212/2)**2       # m^2   
    
    
    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'main_wing'
    
    wing.sref      = 358.25         #
    wing.ar        = 1.83           #
    wing.span      = 25.6           #
    wing.sweep     = 55 * Units.deg #
    wing.symmetric = True           #
    wing.t_c       = 0.03           #
    wing.taper     = 0              # Estimated based on drawing

    # size the wing planform
    SUAVE.Methods.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    wing.chord_mac   = 14.02                 #
    wing.S_exposed   = 0.8*wing.area_wetted  #
    wing.S_affected  = 0.6*wing.area_wetted  #
    wing.e           = 0.74                   # Actual value is unknown
    wing.twist_rc    = 3.0*Units.degrees     #
    wing.twist_tc    = 3.0*Units.degrees     #
    wing.highlift    = False                 #
    #wing.hl          = 1                    #
    #wing.flaps_chord = 20                   #
    #wing.flaps_angle = 20                   #
    #wing.slats_angle = 10                   #
    
    #print wing
    # add to vehicle
    vehicle.append_component(wing)
    
    
    # ------------------------------------------------------------------        
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------        
    # Concorde does not have a horizontal stabilizer
    
    #wing = SUAVE.Components.Wings.Wing()
    #wing.tag = 'horizontal_stabilizer'
    
    #wing.sref      = 32.488         #
    #wing.ar        = 6.16           #
    #wing.span      = 14.146         #
    #wing.sweep     = 30 * Units.deg #
    #wing.symmetric = True           #
    #wing.t_c       = 0.08           #
    #wing.taper     = 0.4            #
    
    ## size the wing planform
    #SUAVE.Methods.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    #wing.chord_mac  = 8.0                   #
    #wing.S_exposed  = 0.8*wing.area_wetted  #
    #wing.S_affected = 0.6*wing.area_wetted  #
    #wing.e          = 0.9                   #
    #wing.twist_rc   = 3.0*Units.degrees     #
    #wing.twist_tc   = 3.0*Units.degrees     #
  
    ## add to vehicle
    #vehicle.append_component(wing)
    
    
    # ------------------------------------------------------------------
    #   Vertcal Stabilizer
    # ------------------------------------------------------------------
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Vertcal Stabilizer'    
    
    wing.sref      = 33.91          #
    wing.ar        = 1.07           # 
    wing.span      = 11.32          #
    wing.sweep     = 55 * Units.deg # Estimate
    wing.symmetric = False          #
    wing.t_c       = 0.04           # Estimate
    wing.taper     = 0.25           # Estimate
    
    # size the wing planform
    SUAVE.Methods.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    wing.chord_mac  = 8.0                   # Estimate
    wing.S_exposed  = 1.0*wing.area_wetted  #
    wing.S_affected = 0.0*wing.area_wetted  #  
    wing.e          = 0.9                   #
    wing.twist_rc   = 0.0*Units.degrees     #
    wing.twist_tc   = 0.0*Units.degrees     #
    wing.vertical   = True    
        
    
        
    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------
    
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'fuselage'
    
    fuselage.num_coach_seats = 0    # Actually ~120, using 0 for length simplicity
    fuselage.seats_abreast   = 4    #
    fuselage.seat_pitch      = 0    #
    fuselage.fineness_nose   = 3.48 #
    fuselage.fineness_tail   = 3.48 #
    fuselage.fwdspace        = 20.83#
    fuselage.aftspace        = 20.83#
    fuselage.width           = 2.87 #
    fuselage.height          = 3.30 #
    
    # size fuselage planform
    SUAVE.Methods.Geometry.Two_Dimensional.Planform.fuselage_planform(fuselage)
    
    # add to vehicle
    vehicle.append_component(fuselage)
    
    
    # ------------------------------------------------------------------
    #  Turbojet
    # ------------------------------------------------------------------    
    
    #turbojet = SUAVE.Components.Propulsors.Turbojet2PASS()
    turbojet = SUAVE.Components.Propulsors.Turbojet_SupersonicPASS()
    turbojet.tag = 'Turbojet Variable Nozzle'
    
    turbojet.propellant = SUAVE.Attributes.Propellants.Jet_A1()
    
    turbojet.analysis_type                 = '1D'     #
    turbojet.diffuser_pressure_ratio       = 1.0      # 1.0 either not known or not relevant
    turbojet.fan_pressure_ratio            = 1.0      #
    turbojet.fan_nozzle_pressure_ratio     = 1.0      #
    turbojet.lpc_pressure_ratio            = 3.1      #
    turbojet.hpc_pressure_ratio            = 5.0      #
    turbojet.burner_pressure_ratio         = 1.0      #
    turbojet.turbine_nozzle_pressure_ratio = 1.0      #
    turbojet.Tt4                           = 1450.0   #
    turbojet.design_thrust                 = 139451.  # 31350 lbs
    turbojet.no_of_engines                 = 4.0      #
    turbojet.engine_length                 = 11.5     # meters - includes 3.4m inlet
    
    # turbojet sizing conditions
    sizing_segment = SUAVE.Components.Propulsors.Segments.Segment()
    
    sizing_segment.M   = 2.04         #
    sizing_segment.alt = 18.0         #
    sizing_segment.T   = 218.0        #
    sizing_segment.p   = 0.239*10**5  # 
    
    # size the turbojet
    turbojet.engine_sizing_1d(sizing_segment)     
    
    # add to vehicle
    vehicle.append_component(turbojet)


    # ------------------------------------------------------------------
    #   Simple Aerodynamics Model
    # ------------------------------------------------------------------ 
    
    aerodynamics = SUAVE.Attributes.Aerodynamics.Fidelity_Zero_Supersonic()
    aerodynamics.initialize(vehicle)
    vehicle.aerodynamics_model = aerodynamics
    
    # ------------------------------------------------------------------
    #   Simple Propulsion Model
    # ------------------------------------------------------------------     
    
    vehicle.propulsion_model = vehicle.Propulsors

    # ------------------------------------------------------------------
    #   Define Configurations
    # ------------------------------------------------------------------

    # --- Takeoff Configuration ---
    config = vehicle.new_configuration("takeoff")
    # this configuration is derived from the baseline vehicle

    # --- Cruise Configuration ---
    config = vehicle.new_configuration("cruise")
    # this configuration is derived from vehicle.Configs.takeoff
    

    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------
    
    return vehicle

#: def define_vehicle()


# ----------------------------------------------------------------------
#   Evaluate the Mission
# ----------------------------------------------------------------------
def evaluate_polar(vehicle):
    
    # ------------------------------------------------------------------    
    #   Run Mission
    # ------------------------------------------------------------------
    n = 7
    altitude = 18000.0 # meters
    mach = np.array([2.02])
    AoA = np.array([[-1.0],[-0.0],[1.0],[2.0],[3.0],[4.0],[5.0]])*np.pi/180
    CL = np.array([[0.0]]*n)
    CD = np.array([[0.0]]*n)
    for ii in range(n):
        atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
        planet = SUAVE.Attributes.Planets.Earth() 
        (p, T, rho, a, mew) = atmosphere.compute_values(altitude)
        konditions = SUAVE.Attributes.Aerodynamics.Conditions_polar()
        konditions.freestream.mach_number = mach
        konditions.freestream.density = rho
        konditions.freestream.viscosity = mew
        konditions.freestream.temperature = T
        konditions.freestream.pressure = p
        konditions.aerodynamics.angle_of_attack = AoA[ii]
        configuration = vehicle.aerodynamics_model.configuration
        geometry = vehicle.aerodynamics_model.geometry
        CL[ii] = SUAVE.Methods.Aerodynamics.Lift.compute_aircraft_lift_supersonic(konditions,configuration,geometry)
        konditions.aerodynamics.lift_coefficient = CL[ii]
        konditions.aerodynamics.drag_breakdown = Data()
        CD[ii] = SUAVE.Methods.Aerodynamics.Drag.compute_aircraft_drag_supersonic(konditions,configuration,geometry)
   
    
    # ------------------------------------------------------------------    
    #   Compute Useful Results
    # ------------------------------------------------------------------
    #SUAVE.Methods.Results.compute_energies(results,summary=True)
    #SUAVE.Methods.Results.compute_efficiencies(results)
    #SUAVE.Methods.Results.compute_velocity_increments(results)
    #SUAVE.Methods.Results.compute_alpha(results)  
    results = (CL,CD,AoA)
    
    return results

# ----------------------------------------------------------------------
#   Plot Results
# ----------------------------------------------------------------------
def post_process(vehicle,results):
    
    fig = plt.figure("Aerodynamic Coefficients")
        
    CL = results[0]
    CD = results[1]
    AoA = results[2]
    #Mc = konditions.freestream.mach_number

    axes = fig.add_subplot(2,1,1)
    axes.plot( CD , CL , 'bo-' )
    axes.set_xlabel('CD')
    axes.set_ylabel('CL')
    axes.grid(True)
    
    axes = fig.add_subplot(2,1,2)
    axes.plot( AoA*180.0/np.pi , CL , 'bo-' )
    axes.set_xlabel('AoA (deg)')
    axes.set_ylabel('CL')
    axes.grid(True)
    
        
    return

# ---------------------------------------------------------------------- 
#   Module Tests
# ----------------------------------------------------------------------

if __name__ == '__main__':
    
    profile_module = False
        
    if not profile_module:
        
        main()
        plt.show()        
        
    else:
        profile_file = 'log_Profile.out'
        
        import cProfile
        cProfile.run('import tut_mission_Boeing_737800 as tut; tut.profile()', profile_file)
        
        import pstats
        p = pstats.Stats(profile_file)
        p.sort_stats('time').print_stats(20)        
        
        import os
        os.remove(profile_file)
    
#: def main()


def profile():
    t0 = time.time()
    vehicle = define_vehicle()
    results = evaluate_polar(vehicle)
    print 'Run Time:' , (time.time()-t0)