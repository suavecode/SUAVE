# Build a Wing
main_wing = Components.Wings.Wing(
    tag = 'main_wing',    
    ref_area     = 1344.0 * Units['sq-ft'],
    aspect_ratio = 10.19,
    # <...>
)

# Assemble a Vehicle
vehicle = SUAVE.Vehicle()
vehicle.add_component(main_wing)
#<...>

# Derive a Configuration
climb_config = vehicle.new_configuration()
config.Wings.Main_Wing.flaps = 'down'

# Plan a Segment
climb = Segments.Climb.Constant_Speed_Constant_Rate( 
    tag = 'Climb',
    altitude_start = 0.0 * Units.km,
    altitude_end   = 5.2 * Units.km,
    analyses = climb_analyses
    #<...>
)

# Create a Mission
mission = Analyses.Missions.Mission()
mission.add_segment(climb)
# <...>


