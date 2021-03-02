## @ingroup Analyses-Noise
# Fidelity_One.py
#
# Created:  
# Modified: Feb 2016, A. Wendorff
# Modified: Oct 2020, M. Clarke

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

from SUAVE.Methods.Noise.Fidelity_One.Propeller.propeller_low_fidelity import propeller_low_fidelity
from SUAVE.Methods.Noise.Fidelity_One.Propeller.propeller_noise_sae    import propeller_noise_sae 

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
        self.harmonics                          = np.empty(shape=[0, 1])
                                                
        settings                                = self.settings
        settings.propeller_SAE_noise_model      = False 
        settings.flyover                        = False    
        settings.approach                       = False
        settings.sideline                       = False
        settings.mic_x_position                 = 0    
        settings.ground_microphone_phi_angles   = np.array([30.,45.,60.,75.,89.9,90.1,105.,120.,135.,150.])*Units.degrees
        settings.ground_microphone_theta_angles = np.array([89.9,89.9,89.9,89.9,89.9,89.9,89.9,89.9, 89.9,89.9 ])*Units.degrees
        settings.center_frequencies             = np.array([16,20,25,31.5,40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, \
                                                            500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
                                                            4000, 5000, 6300, 8000, 10000])        
        settings.lower_frequencies              = np.array([14,18,22.4,28,35.5,45,56,71,90,112,140,180,224,280,355,450,560,710,\
                                                            900,1120,1400,1800,2240,2800,3550,4500,5600,7100,9000 ])
        settings.upper_frequencies              = np.array([18,22.4,28,35.5,45,56,71,90,112,140,180,224,280,355,450,560,710,900,1120,\
                                                            1400,1800,2240,2800,3550,4500,5600,7100,9000,11200 ])
        
        return
            
    def evaluate_noise(self,segment):
        """ Process vehicle to setup geometry, condititon and configuration
    
        Assumptions:
        None
    
        Source:
        N/4
    
        Inputs:
        conditions - DataDict() of aerodynamic conditions
        results    - DataDict() of moment coeffients and stability and body axis derivatives
    
        Outputs:
        None
    
        Properties Used:
        self.geometry
        """         
    
        # unpack
        harmonics     = self.harmonics
        config        = segment.analyses.noise.geometry
        analyses      = segment.analyses
        settings      = self.settings 
        conditions    = segment.state.conditions
        
    
        # unpack 
        alt         = -conditions.frames.inertial.position_vector[:,2]  
        dist        = conditions.frames.inertial.position_vector[:,0] 
        gm_phi      = settings.ground_microphone_phi_angles 
        gm_theta    = settings.ground_microphone_theta_angles 
        cf          = settings.center_frequencies
        
        dim_alt = len(alt)
        num_mic = len(gm_phi)  
        dim_cf  = len(cf)
        
        theta    = np.repeat(np.atleast_2d(gm_theta), dim_alt, axis = 0) 
        phi      = np.repeat(np.atleast_2d(gm_phi), dim_alt, axis = 0) 
        altitude = np.repeat(np.atleast_2d(alt).T, num_mic, axis = 1)
        
        mic_locations        = np.zeros((dim_alt,num_mic,3)) 
        mic_locations[:,:,0] = altitude/np.tan(theta)
        mic_locations[:,:,1] = altitude/np.tan(phi)
        mic_locations[:,:,2] = altitude 
        
        conditions.noise.microphone_phi_angles = gm_phi
        conditions.noise.microphone_locations  = mic_locations
        conditions.noise.number_of_microphones = num_mic
         
        ctrl_pts = len(altitude) 
        
        # create empty arrays for results  
        num_src            = len(config.propulsors) + 1 
        if ('lift_cruise' or 'battery_dual_propeller') in config.propulsors.keys():
            num_src += 1
        source_SPLs_dBA    = np.zeros((ctrl_pts,num_src,num_mic)) 
        source_SPL_spectra = np.zeros((ctrl_pts,num_src,dim_cf ,num_mic))  
        total_SPL_dBA      = np.zeros((ctrl_pts,num_mic))  
        
        # loop for microphone locations 
        for mic_loc in range(num_mic):  
            # calcuate location and geometric angles of noise sources 
            geometric = noise_geometric(segment,analyses,config,mic_loc)
            
            si = 1 
            # make flag to skip if flaps not present        
            if 'flap' in config.wings.main_wing.control_surfaces:            
                airframe_noise                     = noise_airframe_Fink(segment,analyses,config,settings,mic_loc )  
                source_SPLs_dBA[:,si,mic_loc]      = airframe_noise.SPL_dBA          
                source_SPL_spectra[:,si,5:,mic_loc] = airframe_noise.SPL_spectrum
                
            # iterate through sources 
            for source in conditions.noise.sources.keys():  
                for network in config.propulsors.keys():                 
                    if source  == 'turbofan':   
                        if bool(conditions.noise.sources[source].fan) and bool(conditions.noise.sources[source].core): 
                            config.propulsors[source].fan.rotation            = 0 # NEED TO UPDATE ENGINE MODEL WITH FAN SPEED in RPM
                            config.propulsors[source].fan_nozzle.noise_speed  = conditions.noise.sources.turbofan.fan.exit_velocity 
                            config.propulsors[source].core_nozzle.noise_speed = conditions.noise.sources.turbofan.core.exit_velocity 
                            engine_noise                                      = noise_SAE(config.propulsors[source],segment,analyses,config,settings)  
                            source_SPLs_dBA[:,si,mic_loc]                     = engine_noise.SPL_dBA      
                            source_SPL_spectra[:,si,5:,mic_loc]               = engine_noise.SPL_spectrum   
                            
                    elif (source  == 'propeller')  or (source   == 'rotor'): 
                        if bool(conditions.noise.sources[source]) == True : 
                            # Compute Propeller Noise  
                            net           = config.propulsors[network]
                            prop          = config.propulsors[network][source]
                            acoustic_data = conditions.noise.sources[source]  
                            if settings.propeller_SAE_noise_model:
                                propeller_noise                    = propeller_noise_sae(net,prop,acoustic_data,segment,settings,ioprint = 0) 
                                source_SPLs_dBA[:,si,mic_loc]      = propeller_noise.SPL_dBA   
                                source_SPL_spectra[:,si,:,mic_loc] = propeller_noise.SPL_spectrum     
                            else:
                                propeller_noise                    = propeller_low_fidelity(net,prop,acoustic_data,segment,settings,mic_loc,harmonics)  
                                source_SPLs_dBA[:,si,mic_loc]      = propeller_noise.SPL_dBA 
                                source_SPL_spectra[:,si,:,mic_loc] = propeller_noise.SPL_spectrum      
                            
                            si += 1
                
            total_SPL_dBA[:,mic_loc]  = SPL_arithmetic(source_SPLs_dBA[:,:,mic_loc])
        
        conditions.noise.total_SPL_dBA = total_SPL_dBA
        
        return   

