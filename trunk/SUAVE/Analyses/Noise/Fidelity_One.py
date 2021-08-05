## @ingroup Analyses-Noise
# Fidelity_One.py
#
# Created:  
# Modified: Feb 2016, A. Wendorff
# Modified: Apr 2021, M. Clarke
#           Jul 2021, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE 
from SUAVE.Core import Data , Units
from .Noise     import Noise 

# noise imports 
from SUAVE.Methods.Noise.Fidelity_One.Airframe    import noise_airframe_Fink
from SUAVE.Methods.Noise.Fidelity_One.Engine      import noise_SAE 
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import pnl_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import noise_tone_correction
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import epnl_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import noise_certification_limits
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import noise_geometric
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import SPL_arithmetic

from SUAVE.Methods.Noise.Fidelity_One.Propeller.propeller_mid_fidelity import propeller_mid_fidelity 

# package imports
import numpy as np

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------
## @ingroup Analyses-Noise
class Fidelity_One(Noise):
    
    """ SUAVE.Analyses.Noise.Fidelity_One()
    
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
        settings                                 = self.settings
        settings.harmonics                       = np.arange(1,30) 
        settings.flyover                         = False    
        settings.approach                        = False
        settings.sideline                        = False
        settings.print_noise_output              = False 
        settings.mic_x_position                  = 0    
        settings.microphone_array_dimension      = 9
        settings.ground_microphone_phi_angles    = np.linspace(315,225,settings.microphone_array_dimension)*Units.degrees - 1E-8
        settings.ground_microphone_theta_angles  = np.linspace(45,135,settings.microphone_array_dimension)*Units.degrees  + 1E-8
        settings.center_frequencies              = np.array([16,20,25,31.5,40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, \
                                                             500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
                                                             4000, 5000, 6300, 8000, 10000])        
        settings.lower_frequencies               = np.array([14,18,22.4,28,35.5,45,56,71,90,112,140,180,224,280,355,450,560,710,\
                                                             900,1120,1400,1800,2240,2800,3550,4500,5600,7100,9000 ])
        settings.upper_frequencies               = np.array([18,22.4,28,35.5,45,56,71,90,112,140,180,224,280,355,450,560,710,900,1120,\
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
            ground_microphone_phi_angles   - azimuth measured from observer to aircraft body frame     [radians]
            ground_microphone_theta_angles - axial angle measured from observer to aircraft body frame [radians]
            center_frequencies             - 1/3 octave band frequencies                               [unitless]
    
        Outputs:
        None
    
        Properties Used:
        self.geometry
        """         
    
        # unpack 
        config        = segment.analyses.noise.geometry
        analyses      = segment.analyses
        settings      = self.settings 
        conditions    = segment.state.conditions
        print_flag    = settings.print_noise_output
    
        # unpack 
        alt         = -conditions.frames.inertial.position_vector[:,2]   
        gm_phi      = settings.ground_microphone_phi_angles 
        gm_theta    = settings.ground_microphone_theta_angles  
        cf          = settings.center_frequencies
        
        dim_alt   = len(alt)
        dim_phi   = len(gm_phi)  
        dim_theta = len(gm_theta)
        num_mic   = dim_phi*dim_theta
        dim_cf    = len(cf)
        
        # dimension:[control point, theta, phi]
        theta    = np.repeat(np.repeat(np.atleast_2d(gm_theta).T  ,dim_phi  , axis = 1)[np.newaxis,:,:],dim_alt, axis = 0)  
        phi      = np.repeat(np.repeat(np.atleast_2d(gm_phi)      ,dim_theta, axis = 0)[np.newaxis,:,:],dim_alt, axis = 0) 
        altitude = np.repeat(np.repeat(np.atleast_2d(alt).T       ,dim_theta, axis = 1)[:,:,np.newaxis],dim_phi, axis = 2) 
        x_vals   = altitude/np.tan(theta)
        y_vals   = altitude/np.tan(phi)
        z_vals   = altitude   
        
        # store microphone locations 
        mic_locations        = np.zeros((dim_alt,num_mic,3))   
        mic_locations[:,:,0] = x_vals.reshape(dim_alt,num_mic) 
        mic_locations[:,:,1] = y_vals.reshape(dim_alt,num_mic) 
        mic_locations[:,:,2] = z_vals.reshape(dim_alt,num_mic) 
        
        # append microphone locations to conditions
        conditions.noise.microphone_theta_angles = gm_theta
        conditions.noise.microphone_phi_angles   = gm_phi
        conditions.noise.microphone_locations    = mic_locations
        conditions.noise.number_of_microphones   = num_mic
         
        ctrl_pts = len(altitude) 
        
        # create empty arrays for results  
        num_src            = len(config.networks) + 1 
        if ('lift_cruise') in config.networks.keys():
            num_src += 1
        source_SPLs_dBA    = np.zeros((ctrl_pts,num_src,num_mic)) 
        source_SPL_spectra = np.zeros((ctrl_pts,num_src,num_mic,dim_cf ))    
        
        si = 1  
        # iterate through sources 
        for source in conditions.noise.sources.keys():  
            for network in config.networks.keys():                 
                if source  == 'turbofan':
                    geometric = noise_geometric(segment,analyses,config)  
                     
                    # flap noise - only applicable for turbofan aircraft
                    if 'flap' in config.wings.main_wing.control_surfaces:            
                        airframe_noise                = noise_airframe_Fink(segment,analyses,config,settings)  
                        source_SPLs_dBA[:,si,:]       = airframe_noise.SPL_dBA          
                        source_SPL_spectra[:,si,:,5:] = np.repeat(airframe_noise.SPL_spectrum[:,np.newaxis,:], num_mic, axis =1)
                    
                    
                    if bool(conditions.noise.sources[source].fan) and bool(conditions.noise.sources[source].core): 
                                              
                        config.networks[source].fan.rotation            = 0 # FUTURE WORK: NEED TO UPDATE ENGINE MODEL WITH FAN SPEED in RPM
                        config.networks[source].fan_nozzle.noise_speed  = conditions.noise.sources.turbofan.fan.exit_velocity 
                        config.networks[source].core_nozzle.noise_speed = conditions.noise.sources.turbofan.core.exit_velocity
                        engine_noise                                      = noise_SAE(config.networks[source],segment,analyses,config,settings,ioprint = print_flag)  
                        source_SPLs_dBA[:,si,:]                           = np.repeat(np.atleast_2d(engine_noise.SPL_dBA).T, num_mic, axis =1)     # noise measures at one microphone location in segment
                        source_SPL_spectra[:,si,:,5:]                     = np.repeat(engine_noise.SPL_spectrum[:,np.newaxis,:], num_mic, axis =1) # noise measures at one microphone location in segment
                          
                elif (source  == 'propellers')  or (source   == 'rotors'): 
                    if bool(conditions.noise.sources[source]) == True: 
                        net                          = config.networks[network]
                        prop                         = config.networks[network][source]
                        acoustic_data                = conditions.noise.sources[source]   
                        propeller_noise              = propeller_mid_fidelity(net,acoustic_data,segment,settings)  
                        source_SPLs_dBA[:,si,:]      = propeller_noise.SPL_dBA 
                        source_SPL_spectra[:,si,:,:] = propeller_noise.SPL_spectrum    
                           
                        si += 1
             
        conditions.noise.total_SPL_dBA = SPL_arithmetic(source_SPLs_dBA,sum_axis=1)
        
        return   

