# Build a Wing
main_wing = Components.Wings.Wing(
    tag = 'main_wing',    
    ref_area     = 1344.0 , #[sq-ft]
    aspect_ratio = 10.19  , #[-]
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
climb = Missions.Segments.ClimbDescent( 
    tag = 'Climb',
    altitude   = [0.0, 3.0],  # [km]
    config     = climb_config,
    #<...>
)

# Create a Mission
mission = Missions.Mission()
mission.add_segment(climb)
# <...>


