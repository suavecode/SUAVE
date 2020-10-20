## @ingroup Methods-Missions-Segments-Common
# Noise.py
# 
# Created:  Oct 2020, M. Clarke

# ----------------------------------------------------------------------
#  Update Noise
# ----------------------------------------------------------------------
## @ingroup Methods-Missions-Segments-Common
def compute_noise(segment):
    """ Evaluates the energy network to find the thrust force and mass rate

        Inputs -
            segment.analyses.energy_network    [Function]

        Outputs -
            state.conditions:
               frames.body.thrust_force_vector [Newtons]
               weights.vehicle_mass_rate       [kg/s]


        Assumptions -


    """    
    noise_model = segment.analyses.noise
    
    if noise_model:
        noise_model.evaluate_noise(segment)    