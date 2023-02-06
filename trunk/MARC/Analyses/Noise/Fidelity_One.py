## @ingroup Analyses-Noise
# Fidelity_One.py
#
# Created:  
# Modified: Feb 2016, A. Wendorff
# Modified: Apr 2021, M. Clarke
#           Jul 2021, E. Botero
#           Feb 2022, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import MARC 
from MARC.Core import Data , Units
from .Noise     import Noise 

from MARC.Components.Physical_Component import Container 

# noise imports 
from MARC.Methods.Noise.Fidelity_One.Airframe.noise_airframe_Fink                   import noise_airframe_Fink
from MARC.Methods.Noise.Fidelity_One.Engine.noise_SAE                               import noise_SAE  
from MARC.Methods.Noise.Fidelity_One.Noise_Tools.noise_geometric                    import noise_geometric
from MARC.Methods.Noise.Fidelity_One.Noise_Tools.decibel_arithmetic                 import SPL_arithmetic
from MARC.Methods.Noise.Fidelity_One.Noise_Tools.generate_microphone_points         import generate_ground_microphone_points
from MARC.Methods.Noise.Fidelity_One.Noise_Tools.compute_noise_evaluation_locations import compute_ground_noise_evaluation_locations 
from MARC.Methods.Noise.Fidelity_One.Rotor.total_rotor_noise                        import total_rotor_noise 

# package imports
import numpy as np

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------
## @ingroup Analyses-Noise
class Fidelity_One(Noise):
    
    """ MARC.Analyses.Noise.Fidelity_One()
    
        The Fidelity One Noise Analysis Class
        
            Assumptions:
            None
            
            Source:
            N/A
    """
    
    def __defaults__(self):
        
        """ This sets the default values for the analysis.
        
                Assumptions:
                Ground microphone angles start in front of the aircraft (0 deg) and sweep in a lateral direction 
                to the starboard wing and around to the tail (180 deg)
                
                Source:
                N/A
                
                Inputs:
                None
                
                Output:
                None
                
                Properties Used:
                N/A
        """
        
        # Initialize quantities
        settings                                      = self.settings
        settings.harmonics                            = np.arange(1,30) 
        settings.flyover                              = False    
        settings.approach                             = False
        settings.sideline                             = False
        settings.sideline_x_position                  = 0 
        settings.print_noise_output                   = False   
        settings.aircraft_destination_location        = np.array([0,0,0])
        settings.aircraft_departure_location          = np.array([0,0,0])
        
        settings.ground_microphone_locations          = None   
        settings.ground_microphone_x_resolution       = 100
        settings.ground_microphone_y_resolution       = 100
        settings.ground_microphone_x_stencil          = 2
        settings.ground_microphone_y_stencil          = 2
        settings.ground_microphone_min_x              = 1E-6
        settings.ground_microphone_max_x              = 5000 
        settings.ground_microphone_min_y              = 1E-6
        settings.ground_microphone_max_y              = 450        
         
                
        # settings for acoustic frequency resolution
        settings.center_frequencies                   = np.array([16,20,25,31.5,40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, \
                                                                  500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
                                                                  4000, 5000, 6300, 8000, 10000])        
        settings.lower_frequencies                    = np.array([14,18,22.4,28,35.5,45,56,71,90,112,140,180,224,280,355,450,560,710,\
                                                                  900,1120,1400,1800,2240,2800,3550,4500,5600,7100,9000 ])
        settings.upper_frequencies                    = np.array([18,22.4,28,35.5,45,56,71,90,112,140,180,224,280,355,450,560,710,900,1120,\
                                                                 1400,1800,2240,2800,3550,4500,5600,7100,9000,11200 ])
        
        return
            
    def evaluate_noise(self,segment):
        """ Process vehicle to setup geometry, condititon and configuration
    
        Assumptions:
        None
    
        Source:
        N/4
    
        Inputs:
        self.settings.
            center_frequencies  - 1/3 octave band frequencies   [unitless]
    
        Outputs:
        None
    
        Properties Used:
        self.geometry
        """         
    
        # unpack 
        config        = segment.analyses.noise.geometry
        analyses      = segment.analyses
        settings      = self.settings 
        print_flag    = settings.print_noise_output  
        conditions    = segment.state.conditions  
        dim_cf        = len(settings.center_frequencies ) 
        ctrl_pts      = int(segment.state.numerics.number_control_points) 
        
        # generate noise valuation points
        if type(settings.ground_microphone_locations) is not np.ndarray: 
            generate_ground_microphone_points(settings)     
        
        GM_THETA,GM_PHI,REGML,EGML,TGML,num_gm_mic,mic_stencil = compute_ground_noise_evaluation_locations(settings,segment)
          
        # append microphone locations to conditions
        conditions.noise.ground_microphone_theta_angles        = GM_THETA
        conditions.noise.ground_microphone_phi_angles          = GM_PHI
        conditions.noise.ground_microphone_stencil_locations   = mic_stencil        
        conditions.noise.evaluated_ground_microphone_locations = EGML       
        conditions.noise.total_ground_microphone_locations     = TGML
        conditions.noise.number_of_ground_microphones          = num_gm_mic
         
        conditions.noise.total_microphone_theta_angles         = GM_THETA 
        conditions.noise.total_microphone_phi_angles           = GM_PHI 
        conditions.noise.total_microphone_locations            = REGML 
        conditions.noise.total_number_of_microphones           = num_gm_mic 
        
        # create empty arrays for results      
        total_SPL_dBA          = np.ones((ctrl_pts,num_gm_mic))*1E-16 
        total_SPL_spectra      = np.ones((ctrl_pts,num_gm_mic,dim_cf))*1E-16  
         
        # iterate through sources 
        for source in conditions.noise.sources.keys():  
            for network in config.networks.keys(): 
                if source  == 'turbofan': 
                    
                    geometric = noise_geometric(segment,analyses,config)  
                     
                    # flap noise - only applicable for turbofan aircraft
                    if 'flap' in config.wings.main_wing.control_surfaces:   
                
                        source_SPLs_dBA    = np.zeros((ctrl_pts,1,num_gm_mic)) 
                        source_SPL_spectra = np.zeros((ctrl_pts,1,num_gm_mic,dim_cf))
                        
                        airframe_noise               = noise_airframe_Fink(segment,analyses,config,settings)  
                        source_SPLs_dBA[:,0,:]       = airframe_noise.SPL_dBA          
                        source_SPL_spectra[:,0,:,5:] = np.repeat(airframe_noise.SPL_spectrum[:,np.newaxis,:], num_gm_mic , axis =1)
                        
                        # add noise 
                        total_SPL_dBA     = SPL_arithmetic(np.concatenate((total_SPL_dBA[:,None,:],source_SPLs_dBA),axis =1),sum_axis=1)
                        total_SPL_spectra = SPL_arithmetic(np.concatenate((total_SPL_spectra[:,None,:,:],source_SPL_spectra),axis =1),sum_axis=1)
                    
                    
                    if bool(conditions.noise.sources[source].fan) and bool(conditions.noise.sources[source].core): 

                        source_SPLs_dBA    = np.zeros((ctrl_pts,1,num_gm_mic)) 
                        source_SPL_spectra = np.zeros((ctrl_pts,1,num_gm_mic,dim_cf ))
                                                
                                              
                        config.networks[source].fan.rotation             = 0 # FUTURE WORK: NEED TO UPDATE ENGINE MODEL WITH FAN SPEED in RPM
                        config.networks[source].fan_nozzle.noise_speed   = conditions.noise.sources.turbofan.fan.exit_velocity 
                        config.networks[source].core_nozzle.noise_speed  = conditions.noise.sources.turbofan.core.exit_velocity
                        engine_noise                                     = noise_SAE(config.networks[source],segment,analyses,config,settings,ioprint = print_flag)  
                        source_SPLs_dBA[:,0,:]                           = np.repeat(np.atleast_2d(engine_noise.SPL_dBA).T, num_gm_mic , axis =1)     # noise measures at one microphone location in segment
                        source_SPL_spectra[:,0,:,5:]                     = np.repeat(engine_noise.SPL_spectrum[:,np.newaxis,:], num_gm_mic , axis =1) # noise measures at one microphone location in segment
                   
                        # add noise 
                        total_SPL_dBA     = SPL_arithmetic(np.concatenate((total_SPL_dBA[:,None,:],source_SPLs_dBA),axis =1),sum_axis=1)
                        total_SPL_spectra = SPL_arithmetic(np.concatenate((total_SPL_spectra[:,None,:,:],source_SPL_spectra),axis =1),sum_axis=1)
                        
                elif source  == 'rotors':   

                    source_SPLs_dBA    = np.zeros((ctrl_pts,1,num_gm_mic)) 
                    source_SPL_spectra = np.zeros((ctrl_pts,1,num_gm_mic,dim_cf))
                                        
                        
                    if bool(conditions.noise.sources[source]) == True: # if results are present in rotors data structure
                        net                                       = config.networks[network]      # get network  
                        active_groups                             = net.active_propulsor_groups   # determine what groups of propulsors are active
                        unique_rot_groups,identical_rots          = np.unique(net.rotor_group_indexes, return_counts=True)  
                        rotor_noise_sources                       = conditions.noise.sources.rotors 
                        rotor_noise_source_keys                   = list(conditions.noise.sources.rotors.keys())
                        distributed_rotor_noise_SPL_dBA           = np.zeros((sum(active_groups),ctrl_pts,num_gm_mic)) 
                        distributed_rotor_noise_SPL_1_3_spectrum  = np.zeros((sum(active_groups),ctrl_pts,num_gm_mic,dim_cf)) 
                        
                        k_idx = 0
                        for i in range(len(unique_rot_groups)):
                            # if group was active, run rotor noise computation, else skip 
                            if active_groups[i]:    
                                
                                # get stored aeroacoustic data  
                                aeroacoustic_data  = rotor_noise_sources[rotor_noise_source_keys[k_idx]]  
                                
                                # get the rotors associated with aeroaoustic data
                                rotor_group        = Container() 
                                for r_idx , rotor  in enumerate(net.rotors):       
                                    if net.rotor_group_indexes[r_idx] == unique_rot_groups[i]: 
                                        rotor_group.append(rotor)                                  
                                rotor_noise                                      = total_rotor_noise(rotor_group,aeroacoustic_data,segment,settings) 
                                distributed_rotor_noise_SPL_dBA[k_idx]           = rotor_noise.SPL_dBA 
                                distributed_rotor_noise_SPL_1_3_spectrum[k_idx]  = rotor_noise.SPL_1_3_spectrum_dBA                                   
                                k_idx += 1
                                
                        rotor_noise.SPL_dBA          = SPL_arithmetic(distributed_rotor_noise_SPL_dBA ,sum_axis=0)
                        rotor_noise.SPL_1_3_spectrum = SPL_arithmetic(distributed_rotor_noise_SPL_1_3_spectrum ,sum_axis=0)                                
                                
                        source_SPLs_dBA[:,0,:]      = rotor_noise.SPL_dBA 
                        source_SPL_spectra[:,0,:,:] = rotor_noise.SPL_1_3_spectrum 
                        
                        # add noise  
                        total_SPL_dBA     = SPL_arithmetic(np.concatenate((total_SPL_dBA[:,None,:],source_SPLs_dBA),axis =1),sum_axis=1)
                        total_SPL_spectra = SPL_arithmetic(np.concatenate((total_SPL_spectra[:,None,:,:],source_SPL_spectra),axis =1),sum_axis=1)
                            
             
        conditions.noise.total_SPL_dBA              = total_SPL_dBA
        conditions.noise.total_SPL_1_3_spectrum_dBA = total_SPL_spectra
        
        return   

