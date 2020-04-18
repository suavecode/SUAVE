## @ingroup Analyses-Aerodynamics
# Vortex_Lattice.py
#
# Created:  Nov 2013, T. Lukaczyk
# Modified:     2014, T. Lukaczyk, A. Variyar, T. Orra
#           Feb 2016, A. Wendorff
#           Apr 2017, T. MacDonald
#           Nov 2017, E. Botero
#           Dec 2018, M. Clarke
#           Apr 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
import SUAVE

from SUAVE.Core import Data
from SUAVE.Core import Units

from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.VLM import VLM
# local imports
from .Aerodynamics import Aerodynamics
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_vortex_distribution import compute_vortex_distribution
from SUAVE.Plots import plot_vehicle_vlm_panelization  
from SUAVE.Methods.Aerodynamics.Supersonic_Zero.Drag.Cubic_Spline_Blender import Cubic_Spline_Blender

# package imports
import numpy as np 
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline 

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------
## @ingroup Analyses-Aerodynamics
class Vortex_Lattice(Aerodynamics):
    """This builds a surrogate and computes lift using a basic vortex lattice.

    Assumptions:
    None

    Source:
    None
    """ 

    def __defaults__(self):
        """This sets the default values and methods for the analysis.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        N/A
        """  
        self.tag                                     = 'Vortex_Lattice' 
        self.geometry                                = Data()
        self.settings                                = Data()
        self.settings.number_panels_spanwise         = 10
        self.settings.number_panels_chordwise        = 2
        self.settings.vortex_distribution            = Data()   
        
        # conditions table, used for surrogate model training
        self.training                                = Data()     
        self.training.reference_speed_of_sound       = 343.234   
        self.training.angle_of_attack                = np.array([[-5., -2. , 0.0 , 2.0, 5.0 , 8.0, 10.0 , 12.]]).T * Units.deg
        self.training.Mach_subsonic                  = np.array([[0.0, 0.1 , 0.2 , 0.3,  0.5,  0.75 , 0.85 , 0.9]]).T
        self.training.Mach_supersonic                = np.array([[1.2, 1.5, 1.8 , 2.0, 2.25 , 2.5, 3.0, 3.5]]).T            
        self.training.lift_coefficient_sub           = None
        self.training.lift_coefficient_sup           = None
        self.training.wing_lift_coefficient_sub      = None
        self.training.wing_lift_coefficient_sup      = None
        self.training.drag_coefficient_sub           = None
        self.training.drag_coefficient_sup           = None
        self.training.wing_drag_coefficient_sub      = None
        self.training.wing_drag_coefficient_sup      = None

        # surrogoate models
        self.surrogates                              = Data() 
        self.surrogates.lift_coefficient_sub         = None
        self.surrogates.lift_coefficient_sup         = None
        self.surrogates.lift_coefficient_trans       = None
        self.surrogates.wing_lift_coefficient_sub    = None
        self.surrogates.wing_lift_coefficient_sup    = None
        self.surrogates.wing_lift_coefficient_trans  = None
        self.surrogates.drag_coefficient_sub         = None
        self.surrogates.drag_coefficient_sup         = None
        self.surrogates.drag_coefficient_trans       = None
        self.surrogates.wing_drag_coefficient_sub    = None
        self.surrogates.wing_drag_coefficient_sup    = None
        self.surrogates.wing_drag_coefficient_trans  = None 
        
        self.evaluate                                = None
        
    def initialize(self,use_surrogate , vortex_distribution_flag, n_sw , n_cw ):
        """Drives functions to get training samples and build a surrogate.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """                      
        # Unpack:
        geometry = self.geometry
        settings = self.settings      
        
        if n_sw is not None:
            settings.number_panels_spanwise  = n_sw
        
        if n_cw is not None:
            settings.number_panels_chordwise = n_cw
            
        # generate vortex distribution
        VD = compute_vortex_distribution(geometry,settings)      
        
        # Pack
        settings.vortex_distribution = VD
        settings.use_surrogate      = use_surrogate
        
        # Plot vortex discretization of vehicle
        if vortex_distribution_flag == True:
            plot_vehicle_vlm_panelization(VD)        
                
        # If we are using the surrogate
        if use_surrogate == True: 
            # sample training data
            self.sample_training()
                        
            # build surrogate
            self.build_surrogate()        
            
            self.evaluate = self.evaluate_surrogate
               
        else:
            self.evaluate = self.evaluate_no_surrogate


    def evaluate_surrogate(self,state,settings,geometry):
        """Evaluates lift and drag using available surrogates.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        state.conditions.
          freestream.dynamics_pressure       [-]
          angle_of_attack                    [radians]

        Outputs:
        conditions.aerodynamics.lift_breakdown.
          inviscid_wings_lift[wings.*.tag]        [-] CL (wing specific)
          inviscid_wings_lift.total               [-] CL
        conditions.aerodynamics.                  
        inviscid_wings_lift                       [-] CL
        conditions.aerodynamics.drag_breakdown.induced.
          total                                   [-] CDi 
          wings_sectional_drag                    [-] CDiy (wing specific)
          induced.inviscid_wings_drag             [-] CDi (wing specific)
        
        Properties Used:
        self.surrogates.
          lift_coefficient                        [-] CL
          wing_lift_coefficient[wings.*.tag]      [-] CL (wing specific)
        """          
        
        # unpack        
        conditions          = state.conditions
        settings            = self.settings
        geometry            = self.geometry
        surrogates          = self.surrogates
        AoA                 = conditions.aerodynamics.angle_of_attack
        Mach                = conditions.freestream.mach_number 
        
        # Unapck the surrogates
        CL_surrogate_sub          = surrogates.lift_coefficient_sub  
        CL_surrogate_sup          = surrogates.lift_coefficient_sup  
        CL_surrogate_trans        = surrogates.lift_coefficient_trans
        CDi_surrogate_sub         = surrogates.drag_coefficient_sub  
        CDi_surrogate_sup         = surrogates.drag_coefficient_sup  
        CDi_surrogate_trans       = surrogates.drag_coefficient_trans
        wing_CL_surrogates_sub    = surrogates.wing_lift_coefficient_sub  
        wing_CL_surrogates_sup    = surrogates.wing_lift_coefficient_sup  
        wing_CL_surrogates_trans  = surrogates.wing_lift_coefficient_trans
        wing_CDi_surrogates_sub   = surrogates.wing_drag_coefficient_sub  
        wing_CDi_surrogates_sup   = surrogates.wing_drag_coefficient_sup  
        wing_CDi_surrogates_trans = surrogates.wing_drag_coefficient_trans
        
        # Create Result Data Structures         
        data_len                                                           = len(AoA)
        inviscid_lift                                                      = np.zeros([data_len,1]) 
        inviscid_drag                                                      = np.zeros([data_len,1])  
        conditions.aerodynamics.drag_breakdown.induced                     = Data()
        conditions.aerodynamics.drag_breakdown.induced.inviscid_wings_drag = Data()
        conditions.aerodynamics.lift_breakdown                             = Data()
        conditions.aerodynamics.lift_breakdown.inviscid_wings_lift         = Data()
        conditions.aerodynamics.lift_breakdown.compressible_wings          = Data()
        conditions.aerodynamics.drag_breakdown.compressible                = Data()
        inviscid_wings_CLs = Data()
        
        # Spline for Subsonic-to-Transonic-to-Supesonic Regimes
        sub_trans_spline = Cubic_Spline_Blender(0.85,0.95)
        h_sub = lambda M:sub_trans_spline.compute(M)          
        sup_trans_spline = Cubic_Spline_Blender(1.05,1.25)
        h_sup = lambda M:sup_trans_spline.compute(M)          
        
        for i in range(data_len): 
            inviscid_lift[i] = h_sub(Mach[i])*CL_surrogate_sub(AoA[i][0],Mach[i][0])[0]    +\
                              (h_sup(Mach[i]) - h_sub(Mach[i]))*CL_surrogate_trans(AoA[i][0],Mach[i][0])[0] + \
                              (1- h_sup(Mach[i]))*CL_surrogate_sup(AoA[i][0],Mach[i][0])[0]
            
            inviscid_drag[i] = h_sub(Mach[i])*CDi_surrogate_sub(AoA[i][0],Mach[i][0])[0]   +\
                              (h_sup(Mach[i]) - h_sub(Mach[i]))*CDi_surrogate_trans(AoA[i][0],Mach[i][0])[0] + \
                              (1- h_sup(Mach[i]))*CDi_surrogate_sup(AoA[i][0],Mach[i][0])[0]
          
        conditions.aerodynamics.lift_coefficient             = inviscid_lift
        conditions.aerodynamics.lift_breakdown.total         = inviscid_lift 
        conditions.aerodynamics.drag_breakdown.induced.total = inviscid_drag   
        
        for wing in geometry.wings.keys(): 
            inviscid_wing_lifts      = np.zeros([data_len,1])
            inviscid_wing_drags      = np.zeros([data_len,1])            
            for i,_ in enumerate(AoA): 
                inviscid_wing_lifts[i] = h_sub(Mach[i])*wing_CL_surrogates_sub[wing](AoA[i][0],Mach[i][0])[0]    + \
                                         (h_sup(Mach[i]) - h_sub(Mach[i]))*wing_CL_surrogates_trans[wing](AoA[i][0],Mach[i][0])[0] + \
                                         (1- h_sup(Mach[i]))*wing_CL_surrogates_sup[wing](AoA[i][0],Mach[i][0])[0]
               
                inviscid_wing_drags[i] = h_sub(Mach[i])*wing_CDi_surrogates_sub[wing](AoA[i][0],Mach[i][0])[0]  + \
                                         (h_sup(Mach[i]) - h_sub(Mach[i]))*wing_CDi_surrogates_trans[wing](AoA[i][0],Mach[i][0])[0] + \
                                         (1- h_sup(Mach[i]))*wing_CDi_surrogates_sup[wing](AoA[i][0],Mach[i][0])[0]
              
            inviscid_wings_CLs = inviscid_wing_lifts
            conditions.aerodynamics.lift_breakdown.inviscid_wings_lift[wing]          = inviscid_wing_lifts
            conditions.aerodynamics.lift_breakdown.compressible_wings[wing]           = inviscid_wing_lifts      
            conditions.aerodynamics.drag_breakdown.induced.inviscid_wings_drag[wing]  = inviscid_wing_drags
         
        return inviscid_wings_CLs
    
    def evaluate_no_surrogate(self,state,settings,geometry):
        """Evaluates lift and drag directly using VLM
        
        Assumptions:
        no changes to initial geometry or settings
        
        Source:
        N/A
        
        Inputs:
        state.conditions.
          angle_of_attack                         [radians]
          
        Outputs:
        conditions.aerodynamics.lift_breakdown.
          inviscid_wings_lift[wings.*.tag]        [-] CL (wing specific)
          inviscid_wings_lift.total               [-] CL
          inviscid_wings_sectional_lift           [-] Cly  
        conditions.aerodynamics.drag_breakdown.induced.
          total                                   [-] CDi 
          wings_sectional_drag                    [-] CDiy (wing specific)
          induced.inviscid_wings_drag             [-] CDi  (wing specific)        
        conditions.aerodynamics.lift_breakdown. 
          total                                   [-] CDi 
          wings_sectional_lift                    [-] Cly (wing specific)
          induced.inviscid_wings_lift             [-] CDi (wing specific)        
        conditions.aerodynamics.
          pressure_coefficient                    [-] CP
         
        Properties Used:
        self.surrogates.
          lift_coefficient                        [-] CL
          wing_lift_coefficient[wings.*.tag]      [-] CL (wing specific)
        """          
        
        # unpack        
        conditions = state.conditions
        settings   = self.settings
        geometry   = self.geometry     
        
        # Evaluate the VLM
        # if in transonic regime, use surrogate
        inviscid_lift, inviscid_drag, wing_lifts, wing_drags, wing_lift_distribution , wing_drag_distribution , pressure_coefficient = \
            calculate_VLM(conditions,settings,geometry)
        
        # Lift 
        conditions.aerodynamics.lift_coefficient                             = inviscid_lift  
        conditions.aerodynamics.lift_breakdown.total                         = inviscid_lift        
        conditions.aerodynamics.lift_breakdown.compressible_wings            = wing_lifts
        conditions.aerodynamics.lift_breakdown.inviscid_wings_lifts          = wing_lifts
        conditions.aerodynamics.lift_breakdown.inviscid_wings_sectional_lift = wing_lift_distribution
        
        # Drag        
        conditions.aerodynamics.drag_breakdown.induced                       = Data()
        conditions.aerodynamics.drag_breakdown.induced.total                 = inviscid_drag        
        conditions.aerodynamics.drag_breakdown.induced.inviscid_wings_drag   = wing_drags
        conditions.aerodynamics.drag_breakdown.induced.wings_sectional_drag  = wing_drag_distribution 
        
        # Pressure
        conditions.aerodynamics.pressure_coefficient                         = pressure_coefficient
        
        return inviscid_lift
    
    
    def sample_training(self):
        """Call methods to run vortex lattice for sample point evaluation.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        see properties used

        Outputs:
        self.training.
          lift_coefficient            [-] 
          wing_lift_coefficient       [-] (wing specific)
          drag_coefficient            [-]         CL_surrogate_sub               = RectBivariateSpline(AoA_data, mach_data_sub, CL_data_sub)  

          wing_drag_coefficient       [-] (wing specific)

        Properties Used:
        self.geometry.wings.*.tag
        self.settings                 (passed to calculate vortex lattice)
        self.training.angle_of_attack [radians]
        """
        # unpack
        geometry = self.geometry
        settings = self.settings
        training = self.training
        AoA      = training.angle_of_attack
        Mach_sub = training.Mach_subsonic
        Mach_sup = training.Mach_supersonic
        a        = self.training.reference_speed_of_sound
        
        # Setup Konditions                      
        konditions                              = Data()
        konditions.aerodynamics                 = Data()
        konditions.freestream                   = Data()
        konditions.aerodynamics.angle_of_attack = AoA 
        
        
        # Assign placeholders        
        CL_sub    = np.zeros((len(AoA),len(Mach_sub)))
        CL_sup    = np.zeros_like(CL_sub)  
        CDi_sub   = np.zeros_like(CL_sub)
        CDi_sup   = np.zeros_like(CL_sub)
        CL_w_sub  = Data()
        CL_w_sup  = Data()
        CDi_w_sub = Data()
        CDi_w_sup = Data() 
        for wing in geometry.wings.keys():
            CL_w_sub[wing]  = np.zeros_like(CL_sub)
            CL_w_sup[wing]  = np.zeros_like(CL_sub)
            CDi_w_sub[wing] = np.zeros_like(CL_sub)
            CDi_w_sup[wing] = np.zeros_like(CL_sub)
                
        # Get the training data 
        count = 0
        for mach_sub, mach_sup in zip(Mach_sub, Mach_sup):
            konditions.freestream.mach_number = mach_sub*np.ones_like(AoA) 
            konditions.freestream.velocity    = mach_sub*a* np.ones_like(AoA) 
            total_lift_sub, total_drag_sub, wing_lifts_sub, wing_drags_sub , wing_lift_distribution_sub , wing_drag_distribution_sub , pressure_coefficient_sub = \
                calculate_VLM(konditions,settings,geometry)
            
            konditions.freestream.mach_number = mach_sup*np.ones_like(AoA)
            konditions.freestream.velocity    = mach_sup*a*np.ones_like(AoA)  
            total_lift_sup, total_drag_sup, wing_lifts_sup, wing_drags_sup , wing_lift_distribution_sup , wing_drag_distribution_sup , pressure_coefficient_sup = \
                calculate_VLM(konditions,settings,geometry)
            
            # store training data
            CL_sub[:,count]   = total_lift_sub[:,0]
            CL_sup[:,count]   = total_lift_sup[:,0]
            CDi_sub[:,count]  = total_drag_sub[:,0]                
            CDi_sup[:,count]  = total_drag_sup[:,0]           
            for wing in geometry.wings.keys():
                CL_w_sub[wing][:,count]    = wing_lifts_sub[wing][:,0]
                CL_w_sup[wing][:,count]    = wing_lifts_sup[wing][:,0]
                CDi_w_sub[wing][:,count]   = wing_drags_sub[wing][:,0]                 
                CDi_w_sup[wing][:,count]   = wing_drags_sup[wing][:,0]                
            
            count += 1 
            
        # surrogate not run on sectional coefficients and pressure coefficients
        # Store training data 
        training.lift_coefficient_sub         = CL_sub
        training.lift_coefficient_sup         = CL_sup
        training.wing_lift_coefficient_sub    = CL_w_sub        
        training.wing_lift_coefficient_sup    = CL_w_sup
        training.drag_coefficient_sub         = CDi_sub
        training.drag_coefficient_sup         = CDi_sup
        training.wing_drag_coefficient_sub    = CDi_w_sub        
        training.wing_drag_coefficient_sup    = CDi_w_sup
        
        return
        
    def build_surrogate(self):
        """Build a surrogate using sample evaluation results.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        see properties used

        Outputs:
        self.surrogates.
          lift_coefficient            
          wing_lift_coefficient       
          drag_coefficient            
          wing_drag_coefficient

        Properties Used:
        self.training.
          lift_coefficient            [-] 
          wing_lift_coefficient       [-] (wing specific)
          drag_coefficient            [-] 
          wing_drag_coefficient       [-] (wing specific)
        """           

        # unpack data
        surrogates     = self.surrogates
        training       = self.training
        geometry       = self.geometry
        AoA_data       = training.angle_of_attack[:,0]
        mach_data_sub  = training.Mach_subsonic[:,0] 
        mach_data_sup  = training.Mach_supersonic[:,0]
        CL_data_sub    = training.lift_coefficient_sub   
        CL_data_sup    = training.lift_coefficient_sup      
        CDi_data_sub   = training.drag_coefficient_sub         
        CDi_data_sup   = training.drag_coefficient_sup 
        CL_w_data_sub  = training.wing_lift_coefficient_sub
        CL_w_data_sup  = training.wing_lift_coefficient_sup     
        CDi_w_data_sub = training.wing_drag_coefficient_sub         
        CDi_w_data_sup = training.wing_drag_coefficient_sup 
         
        # transonic regime   	                             
        CL_data_trans        = np.zeros((len(mach_data_sub),3))	      
        CDi_data_trans       = np.zeros((len(mach_data_sub),2))	 	      
        CL_w_data_trans      = Data()	                     
        CDi_w_data_trans     = Data()    
        CL_data_trans[:,0]   = CL_data_sub[:,-1]    	     
        CL_data_trans[:,1]   = CL_data_sup[:,0] 
        CL_data_trans[:,2]   = CL_data_sup[:,1] 
        CDi_data_trans[:,0]  = CDi_data_sub[:,-1]	     
        CDi_data_trans[:,1]  = CDi_data_sup[:,0] 
        
        mach_data_trans_CL       = np.array([mach_data_sub[-1],mach_data_sup[0],mach_data_sup[1]]) 
        mach_data_trans_CDi      = np.array([mach_data_sub[-1],mach_data_sup[0]]) 
         
        for wing in geometry.wings.keys():	        
            CLw                    = np.zeros_like(CL_data_trans)
            CDiw                   = np.zeros_like(CDi_data_trans)            
            CLw[:,0]               = CL_w_data_sub[wing][:,-1]   	 
            CLw[:,1]               = CL_w_data_sup[wing][:,0]  	
            CLw[:,2]               = CL_w_data_sup[wing][:,1]  	
            CDiw[:,0]              = CDi_w_data_sub[wing][:,-1]    
            CDiw[:,1]              = CDi_w_data_sup[wing][:,0]   
            CL_w_data_trans[wing]  = CLw
            CDi_w_data_trans[wing] = CDiw 
         
        CL_surrogate_sub               = RectBivariateSpline(AoA_data, mach_data_sub, CL_data_sub)  
        CL_surrogate_sup               = RectBivariateSpline(AoA_data, mach_data_sup, CL_data_sup) 
        CL_surrogate_trans             = interp2d(AoA_data, mach_data_trans_CL, CL_data_trans.T,kind = 'linear')  
        CDi_surrogate_sub              = RectBivariateSpline(AoA_data, mach_data_sub, CDi_data_sub)  
        CDi_surrogate_sup              = RectBivariateSpline(AoA_data, mach_data_sup, CDi_data_sup)    
        CDi_surrogate_trans            = interp2d(AoA_data, mach_data_trans_CDi, CDi_data_trans.T, kind = 'linear')  
        
        CL_w_surrogates_sub            = Data() 
        CL_w_surrogates_sup            = Data() 
        CL_w_surrogates_trans          = Data() 
        CDi_w_surrogates_sub           = Data()             
        CDi_w_surrogates_sup           = Data() 
        CDi_w_surrogates_trans         = Data()
        
        for wing in geometry.wings.keys():    
            CL_w_surrogates_sub[wing]    = RectBivariateSpline(AoA_data, mach_data_sub, CL_w_data_sub[wing]) 
            CL_w_surrogates_sup[wing]    = RectBivariateSpline(AoA_data, mach_data_sub, CL_w_data_sup[wing])   
            CL_w_surrogates_trans[wing]  = interp2d(AoA_data, mach_data_trans_CL, CL_w_data_trans[wing].T,kind = 'linear') 
            CDi_w_surrogates_sub[wing]   = RectBivariateSpline(AoA_data, mach_data_sub, CDi_w_data_sub[wing])            
            CDi_w_surrogates_sup[wing]   = RectBivariateSpline(AoA_data, mach_data_sub, CDi_w_data_sup[wing])  
            CDi_w_surrogates_trans[wing] = interp2d(AoA_data, mach_data_trans_CDi, CDi_w_data_trans[wing].T,kind = 'linear')           
    
        # Pack the outputs
        surrogates.lift_coefficient_sub        = CL_surrogate_sub  
        surrogates.lift_coefficient_sup        = CL_surrogate_sup  
        surrogates.lift_coefficient_trans      = CL_surrogate_trans
        surrogates.wing_lift_coefficient_sub   = CL_w_surrogates_sub  
        surrogates.wing_lift_coefficient_sup   = CL_w_surrogates_sup  
        surrogates.wing_lift_coefficient_trans = CL_w_surrogates_trans
        surrogates.drag_coefficient_sub        = CDi_surrogate_sub  
        surrogates.drag_coefficient_sup        = CDi_surrogate_sup  
        surrogates.drag_coefficient_trans      = CDi_surrogate_trans
        surrogates.wing_drag_coefficient_sub   = CDi_w_surrogates_sub  
        surrogates.wing_drag_coefficient_sup   = CDi_w_surrogates_sup  
        surrogates.wing_drag_coefficient_trans = CDi_w_surrogates_trans
        
        return

# ----------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------
def calculate_VLM(conditions,settings,geometry):
    """Calculate the total vehicle lift coefficient and specific wing coefficients (with specific wing reference areas)
    using a vortex lattice method.

    Assumptions:
    None

    Source:
    N/A

    Inputs:
    conditions                      (passed to vortex lattice method)
    settings                        (passed to vortex lattice method)
    geometry.reference_area         [m^2]
    geometry.wings.*.reference_area (each wing is also passed to the vortex lattice method)

    Outputs:
    total_lift_coeff                [array]
    total_induced_drag_coeff        [array]
    wing_lifts                      [Data]
    wing_drags                      [Data]
    Properties Used:
    
    """            
    # iterate over wings
    total_lift_coeff = 0.0
    wing_lifts = Data()
    wing_drags = Data()
    
    total_lift_coeff,total_induced_drag_coeff, CM, CL_wing, CDi_wing, cl_y , cdi_y , CPi = VLM(conditions,settings,geometry)

    i = 0
    for wing in geometry.wings.values():
        wing_lifts[wing.tag] = 1*(np.atleast_2d(CL_wing[:,i]).T)
        wing_drags[wing.tag] = 1*(np.atleast_2d(CDi_wing[:,i]).T)
        i+=1
        if wing.symmetric:
            i+=1

    return total_lift_coeff, total_induced_drag_coeff, wing_lifts, wing_drags , cl_y , cdi_y , CPi
