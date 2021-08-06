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
#           Jun 2020, E. Botero
#           Sep 2020, M. Clarke 
#           May 2021, E. Botero
#           Jun 2021, R. Erhard

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
from SUAVE.Methods.Aerodynamics.Supersonic_Zero.Drag.Cubic_Spline_Blender import Cubic_Spline_Blender

# package imports
import numpy as np 
from scipy.interpolate import interp2d, RectBivariateSpline, RegularGridInterpolator

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
        self.tag                                      = 'Vortex_Lattice' 
        self.geometry                                 = Data()
        self.settings                                 = Data()
        self.settings.number_spanwise_vortices        = 15
        self.settings.number_chordwise_vortices       = 5
        self.settings.wing_spanwise_vortices          = None
        self.settings.wing_chordwise_vortices         = None
        self.settings.fuselage_spanwise_vortices      = None
        self.settings.fuselage_chordwise_vortices     = None 
        
        self.settings.spanwise_cosine_spacing         = True
        self.settings.vortex_distribution             = Data()   
        self.settings.model_fuselage                  = False
        self.settings.leading_edge_suction_multiplier = 1.0
        self.settings.initial_timestep_offset         = 0
        self.settings.wake_development_time           = 0.05
        self.settings.number_of_wake_timesteps        = 30
        self.settings.propeller_wake_model            = False
        self.settings.use_bemt_wake_model             = False
        self.settings.discretize_control_surfaces     = False
        self.settings.use_VORLAX_matrix_calculation   = False
        self.settings.floating_point_precision        = np.float32
        self.settings.use_surrogate                   = True

        # conditions table, used for surrogate model training
        self.training                                = Data()
        self.training.angle_of_attack                = np.array([[-5., -2. , 0.0 , 2.0, 5.0, 8.0, 10.0 , 12., 45., 75.]]).T * Units.deg 
        self.training.Mach                           = np.array([[0.0, 0.1  , 0.2 , 0.3,  0.5,  0.75 , 0.85 , 0.9,\
                                                                  1.3, 1.35 , 1.5 , 2.0, 2.25 , 2.5  , 3.0  , 3.5]]).T          


        self.training.lift_coefficient_sub           = None
        self.training.lift_coefficient_sup           = None
        self.training.wing_lift_coefficient_sub      = None
        self.training.wing_lift_coefficient_sup      = None
        self.training.drag_coefficient_sub           = None
        self.training.drag_coefficient_sup           = None
        self.training.wing_drag_coefficient_sub      = None
        self.training.wing_drag_coefficient_sup      = None
        
        # blending function 
        self.hsub_min                                = 0.85
        self.hsub_max                                = 0.95
        self.hsup_min                                = 1.05
        self.hsup_max                                = 1.25 

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
        
    def initialize(self,use_surrogate,n_sw,n_cw,propeller_wake_model, use_bemt_wake_model,ito,wdt,nwts,mf):
        """Drives functions to get training samples and build a surrogate.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        use_surrogate                                       [bool]
        n_sw                   number of spanwise vortices  [int]
        n_cw                   number of chordwise vortices [int]
        propeller_wake_model                                [bool] 
        ito                    initial timestep offset      [s]            
        wdt                    wake development time        [s]
        nwts                   number of wake timesteps     [int]

        Outputs:
        None

        Properties Used:
        None
        """                      
        # Unpack:
        settings = self.settings      
        
        if n_sw is not None:
            settings.number_spanwise_vortices  = n_sw
        
        if n_cw is not None:
            settings.number_chordwise_vortices = n_cw 
            
        settings.use_surrogate              = use_surrogate
        settings.propeller_wake_model       = propeller_wake_model 
        settings.use_bemt_wake_model        = use_bemt_wake_model
        settings.initial_timestep_offset    = ito
        settings.wake_development_time      = wdt
        settings.number_of_wake_timesteps   = nwts
        settings.model_fuselage             = mf
        
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
          inviscid_wings[wings.*.tag]             [-] CL (wing specific)
          inviscid_wings_lift.total               [-] CL
          compressible_wing                       [-] CL (wing specific)
        conditions.aerodynamics.lift_coefficient  [-] CL
        conditions.aerodynamics.drag_breakdown.induced.
          total                                   [-] CDi 
          inviscid                                [-] CDi 
          wings_sectional_drag                    [-] CDiy (wing specific)
          inviscid_wings                          [-] CDi (wing specific)
          
        Properties Used:
        self.surrogates.
          lift_coefficient                        [-] CL
          wing_lift_coefficient[wings.*.tag]      [-] CL (wing specific)
        """          
        
        # unpack        
        conditions  = state.conditions
        settings    = self.settings
        geometry    = self.geometry
        surrogates  = self.surrogates
        hsub_min    = self.hsub_min
        hsub_max    = self.hsub_max
        hsup_min    = self.hsup_min
        hsup_max    = self.hsup_max 

        AoA         = conditions.aerodynamics.angle_of_attack.T[0]
        Mach        = conditions.freestream.mach_number.T[0]
        
        
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
        conditions.aerodynamics.drag_breakdown.induced.inviscid_wings      = Data()
        conditions.aerodynamics.lift_breakdown                             = Data()
        conditions.aerodynamics.lift_breakdown.inviscid_wings              = Data()
        conditions.aerodynamics.lift_breakdown.compressible_wings          = Data()
        conditions.aerodynamics.drag_breakdown.compressible                = Data() 
        
        # 3 Cases for surrogates, subsonic only, supersonic only, and both
        if CL_surrogate_sup == None:
            inviscid_lift = CL_surrogate_sub(AoA,Mach,grid=False)
            inviscid_drag = CDi_surrogate_sub(AoA,Mach,grid=False)
        elif CL_surrogate_sub == None:
            inviscid_lift = CL_surrogate_sup(AoA,Mach,grid=False)
            inviscid_drag = CDi_surrogate_sup(AoA,Mach,grid=False)     
        else:
            # Spline for Subsonic-to-Transonic-to-Supersonic Regimes
            sub_trans_spline = Cubic_Spline_Blender(hsub_min,hsub_max)
            h_sub            = lambda M:sub_trans_spline.compute(M)          
            sup_trans_spline = Cubic_Spline_Blender(hsup_min,hsup_max) 
            h_sup            = lambda M:sup_trans_spline.compute(M)          
        
            inviscid_lift = h_sub(Mach)*CL_surrogate_sub(AoA,Mach,grid=False)    +\
                              (h_sup(Mach) - h_sub(Mach))*CL_surrogate_trans((AoA,Mach))+ \
                              (1- h_sup(Mach))*CL_surrogate_sup(AoA,Mach,grid=False)
    
            inviscid_drag = h_sub(Mach)*CDi_surrogate_sub(AoA,Mach,grid=False)   +\
                              (h_sup(Mach) - h_sub(Mach))*CDi_surrogate_trans((AoA,Mach))+ \
                              (1- h_sup(Mach))*CDi_surrogate_sup(AoA,Mach,grid=False)
    
        # Pack
        conditions.aerodynamics.lift_coefficient                = np.atleast_2d(inviscid_lift).T
        conditions.aerodynamics.lift_breakdown.total            = np.atleast_2d(inviscid_lift).T
        conditions.aerodynamics.drag_breakdown.induced.inviscid = np.atleast_2d(inviscid_drag).T
        
        for wing in geometry.wings.keys(): 
            
            if CL_surrogate_sup == None:
                inviscid_wing_lifts = wing_CL_surrogates_sub[wing](AoA,Mach,grid=False)
                inviscid_wing_drags = wing_CDi_surrogates_sub[wing](AoA,Mach,grid=False)            
                
            elif CL_surrogate_sub == None:
                inviscid_wing_lifts = wing_CL_surrogates_sup[wing](AoA,Mach,grid=False)
                inviscid_wing_drags = wing_CDi_surrogates_sup[wing](AoA,Mach,grid=False)                
            
            else:
                inviscid_wing_lifts = h_sub(Mach)*wing_CL_surrogates_sub[wing](AoA,Mach,grid=False)    + \
                                        (h_sup(Mach) - h_sub(Mach))*wing_CL_surrogates_trans[wing]((AoA,Mach))+ \
                                        (1- h_sup(Mach))*wing_CL_surrogates_sup[wing](AoA,Mach,grid=False)
                
                inviscid_wing_drags = h_sub(Mach)*wing_CDi_surrogates_sub[wing](AoA,Mach,grid=False)  + \
                                        (h_sup(Mach) - h_sub(Mach))*wing_CDi_surrogates_trans[wing]((AoA,Mach))+ \
                                        (1- h_sup(Mach))*wing_CDi_surrogates_sup[wing](AoA,Mach,grid=False)
             
            # Pack 
            conditions.aerodynamics.lift_breakdown.inviscid_wings[wing]         = np.atleast_2d(inviscid_wing_lifts).T
            conditions.aerodynamics.lift_breakdown.compressible_wings[wing]     = np.atleast_2d(inviscid_wing_lifts).T
            conditions.aerodynamics.drag_breakdown.induced.inviscid_wings[wing] = np.atleast_2d(inviscid_wing_drags).T
         
        return     
    
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
          inviscid_wings_sectional                [-] Cly  
          compressible_wing                       [-] CL (wing specific)
        conditions.aerodynamics.drag_breakdown.induced.
          total                                   [-] CDi 
          inviscid                                [-] CDi 
          wings_sectional_drag                    [-] CDiy (wing specific)
          induced.inviscid_wings                  [-] CDi  (wing specific)        
    
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
        inviscid_lift, inviscid_drag, wing_lifts, wing_drags, wing_lift_distribution, \
        wing_drag_distribution, induced_angle_distribution, pressure_coefficient = \
            calculate_VLM(conditions,settings,geometry)
        
        # Lift 
        conditions.aerodynamics.lift_coefficient                        = inviscid_lift  
        conditions.aerodynamics.lift_breakdown.total                    = inviscid_lift        
        conditions.aerodynamics.lift_breakdown.compressible_wings       = wing_lifts
        conditions.aerodynamics.lift_breakdown.inviscid_wings           = wing_lifts
        conditions.aerodynamics.lift_breakdown.inviscid_wings_sectional = wing_lift_distribution
        
        # Drag        
        conditions.aerodynamics.drag_breakdown.induced                 = Data()
        conditions.aerodynamics.drag_breakdown.induced.total           = inviscid_drag     
        conditions.aerodynamics.drag_breakdown.induced.inviscid        = inviscid_drag     
        conditions.aerodynamics.drag_breakdown.induced.inviscid_wings  = wing_drags
        conditions.aerodynamics.drag_breakdown.induced.wings_sectional = wing_drag_distribution 
        conditions.aerodynamics.drag_breakdown.induced.angle           = induced_angle_distribution
        
        # Pressure
        conditions.aerodynamics.pressure_coefficient                   = pressure_coefficient
        
        return  
    
    
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
          drag_coefficient            [-]          

          wing_drag_coefficient       [-] (wing specific)

        Properties Used:
        self.geometry.wings.*.tag
        self.settings                 (passed to calculate vortex lattice)
        self.training.angle_of_attack [radians]
        """
        # unpack
        geometry      = self.geometry
        settings      = self.settings
        training      = self.training
        AoA           = training.angle_of_attack 
        Mach          = training.Mach
        lenAoA        = len(AoA)
        sub_len       = int(sum(Mach<1.))
        sup_len       = len(Mach)-sub_len
        
        # Assign placeholders        
        CL_sub    = np.zeros((lenAoA,sub_len))
        CL_sup    = np.zeros((lenAoA,sup_len))
        CDi_sub   = np.zeros((lenAoA,sub_len))
        CDi_sup   = np.zeros((lenAoA,sup_len))
        CL_w_sub  = Data()
        CL_w_sup  = Data()
        CDi_w_sub = Data()
        CDi_w_sup = Data() 
            
        # Setup new array shapes for vectorization
        lenM   = len(Mach)
        AoAs   = np.atleast_2d(np.tile(AoA,lenM).T.flatten()).T
        Machs  = np.atleast_2d(np.tile(Mach,lenAoA).flatten()).T
        zeros  = np.zeros_like(Machs)
        
        # Setup Konditions    
        konditions                              = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
        konditions.aerodynamics.angle_of_attack = AoAs
        konditions.freestream.mach_number       = Machs
        konditions.freestream.velocity          = zeros
        
        total_lift, total_drag, wing_lifts, wing_drags, _, _, _, _ = calculate_VLM(konditions,settings,geometry)     
    
        # Split subsonic from supersonic
        if np.sum(Machs<1.)==0:
            sub_sup_split = 0
        else:
            sub_sup_split = np.where(Machs < 1.0)[0][-1] + 1 
        len_sub_mach  = np.sum(Mach<1.)
        len_sup_mach  = lenM - len_sub_mach
        
        # Divide up the data to get ready to store
        CL_sub  = total_lift[0:sub_sup_split,0]
        CL_sup  = total_lift[sub_sup_split:,0]
        CDi_sub = total_drag[0:sub_sup_split,0]
        CDi_sup = total_drag[sub_sup_split:,0]
        
        # A little reshape to get into the right order
        CL_sub  = np.reshape(CL_sub,(len_sub_mach,lenAoA)).T
        CL_sup  = np.reshape(CL_sup,(len_sup_mach,lenAoA)).T
        CDi_sub = np.reshape(CDi_sub,(len_sub_mach,lenAoA)).T
        CDi_sup = np.reshape(CDi_sup,(len_sup_mach,lenAoA)).T
        
        # Now do the same for each wing
        for wing in geometry.wings.keys():
            
            # Slice out the sub and supersonic
            CL_wing_sub  = wing_lifts[wing][0:sub_sup_split,0]
            CL_wing_sup  = wing_lifts[wing][sub_sup_split:,0]
            CDi_wing_sub = wing_drags[wing][0:sub_sup_split,0]  
            CDi_wing_sup = wing_drags[wing][sub_sup_split:,0]  
            
            # Rearrange and pack
            CL_w_sub[wing]  = np.reshape(CL_wing_sub,(len_sub_mach,lenAoA)).T
            CL_w_sup[wing]  = np.reshape(CL_wing_sup,(len_sup_mach,lenAoA)).T
            CDi_w_sub[wing] = np.reshape(CDi_wing_sub,(len_sub_mach,lenAoA)).T   
            CDi_w_sup[wing] = np.reshape(CDi_wing_sup,(len_sup_mach,lenAoA)).T   
        
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
        Mach           = training.Mach
        AoA_data       = training.angle_of_attack[:,0]
        if np.sum(Mach<1.)==0:
            sub_sup_split = 0
        else:
            sub_sup_split = np.where(Mach < 1.0)[0][-1] + 1 
        mach_data_sub  = training.Mach[0:sub_sup_split,0]
        mach_data_sup  = training.Mach[sub_sup_split:,0]
        CL_data_sub    = training.lift_coefficient_sub   
        CL_data_sup    = training.lift_coefficient_sup
        CDi_data_sub   = training.drag_coefficient_sub         
        CDi_data_sup   = training.drag_coefficient_sup 
        CL_w_data_sub  = training.wing_lift_coefficient_sub
        CL_w_data_sup  = training.wing_lift_coefficient_sup     
        CDi_w_data_sub = training.wing_drag_coefficient_sub         
        CDi_w_data_sup = training.wing_drag_coefficient_sup
        
        # Instantiate surrogates as none for now
        CL_surrogate_sub        = None
        CL_surrogate_sup        = None
        CL_surrogate_trans      = None
        CL_w_surrogates_sub     = None
        CL_w_surrogates_sup     = None
        CL_w_surrogates_trans   = None
        CDi_surrogate_sub       = None
        CDi_surrogate_sup       = None
        CDi_surrogate_trans     = None
        CDi_w_surrogates_sub    = None
        CDi_w_surrogates_sup    = None
        CDi_w_surrogates_trans  = None
        

        # Check for 3 different cases, pure subsonic, pure supersonic, both
        SUB = np.shape(CL_data_sub)[1]>0
        SUP = np.shape(CL_data_sup)[1]>0

        # Do the subsonic surrogates:
        if SUB:
            CL_surrogate_sub  = RectBivariateSpline(AoA_data, mach_data_sub, CL_data_sub)
            CDi_surrogate_sub = RectBivariateSpline(AoA_data, mach_data_sub, CDi_data_sub)  
        
        # Do the supersonic surrogates:
        if SUP:
            CL_surrogate_sup  = RectBivariateSpline(AoA_data, mach_data_sup, CL_data_sup) 
            CDi_surrogate_sup = RectBivariateSpline(AoA_data, mach_data_sup, CDi_data_sup)    
                
        # Do the transonic regime surrogates:
        if SUB and SUP:	                             
            CL_data_trans        = np.zeros((len(AoA_data),3))	      
            CDi_data_trans       = np.zeros((len(AoA_data),3))	 	      
            CL_w_data_trans      = Data()	                     
            CDi_w_data_trans     = Data()    
            CL_data_trans[:,0]   = CL_data_sub[:,-1]    	     
            CL_data_trans[:,1]   = CL_data_sup[:,0] 
            CL_data_trans[:,2]   = CL_data_sup[:,1] 
            CDi_data_trans[:,0]  = CDi_data_sub[:,-1]	     
            CDi_data_trans[:,1]  = CDi_data_sup[:,0] 
    
            mach_data_trans_CL   = np.array([mach_data_sub[-1],mach_data_sup[0],mach_data_sup[1]]) 
            mach_data_trans_CDi  = np.array([mach_data_sub[-1],mach_data_sup[0],mach_data_sup[1]]) 
    
            CL_surrogate_trans  = RegularGridInterpolator((AoA_data, mach_data_trans_CL), CL_data_trans, \
                                                                     method = 'linear', bounds_error=False, fill_value=None)  
            CDi_surrogate_trans = RegularGridInterpolator((AoA_data, mach_data_trans_CDi), CDi_data_trans, \
                                                                     method = 'linear', bounds_error=False, fill_value=None)  

        CL_w_surrogates_sub            = Data() 
        CL_w_surrogates_sup            = Data() 
        CL_w_surrogates_trans          = Data() 
        CDi_w_surrogates_sub           = Data()             
        CDi_w_surrogates_sup           = Data() 
        CDi_w_surrogates_trans         = Data()
        
        for wing in geometry.wings.keys():           
            
            # Do the subsonic surrogates:
            if SUB:
                CL_w_surrogates_sub[wing]    = RectBivariateSpline(AoA_data, mach_data_sub, CL_w_data_sub[wing]) 
                CDi_w_surrogates_sub[wing]   = RectBivariateSpline(AoA_data, mach_data_sub, CDi_w_data_sub[wing])  
            # Do the supersonic surrogates:
            if SUP:
                CL_w_surrogates_sup[wing]    = RectBivariateSpline(AoA_data, mach_data_sup, CL_w_data_sup[wing]) 
                CDi_w_surrogates_sup[wing]   = RectBivariateSpline(AoA_data, mach_data_sup, CDi_w_data_sup[wing])
            # Do the transonic regime surrogates:
            if SUB and SUP:	  
                CLw                    = np.zeros_like(CL_data_trans)
                CDiw                   = np.zeros_like(CDi_data_trans)            
                CLw[:,0]               = CL_w_data_sub[wing][:,-1]   	 
                CLw[:,1]               = CL_w_data_sup[wing][:,0]  	
                CLw[:,2]               = CL_w_data_sup[wing][:,1]  	
                CDiw[:,0]              = CDi_w_data_sub[wing][:,-1]    
                CDiw[:,1]              = CDi_w_data_sup[wing][:,0]   
                CDiw[:,2]              = CDi_w_data_sup[wing][:,1]   
                CL_w_data_trans[wing]  = CLw
                CDi_w_data_trans[wing] = CDiw  
                
                CL_w_surrogates_trans[wing]  = RegularGridInterpolator((AoA_data, mach_data_trans_CL),\
                                                                       CL_w_data_trans[wing],method = 'linear', \
                                                                       bounds_error=False, fill_value=None)     
                CDi_w_surrogates_trans[wing] = RegularGridInterpolator((AoA_data, mach_data_trans_CL), \
                                                                       CDi_w_data_trans[wing],method = 'linear', \
                                                                       bounds_error=False, fill_value=None)           
    
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
    total_lift_coeff   = 0.0
    wing_lifts         = Data()
    wing_drags         = Data()
    wing_induced_angle = Data()
        
    results = VLM(conditions,settings,geometry)
    total_lift_coeff          = results.CL
    total_induced_drag_coeff  = results.CDi
    CL_wing                   = results.CL_wing  
    CDi_wing                  = results.CDi_wing 
    cl_y                      = results.cl_y     
    cdi_y                     = results.cdi_y    
    alpha_i                   = results.alpha_i  
    CPi                       = results.CP      
    
    # Dimensionalize the lift and drag for each wing
    areas = geometry.vortex_distribution.wing_areas
    dim_wing_lifts = CL_wing  * areas
    dim_wing_drags = CDi_wing * areas
    
    i = 0
    # Assign the lift and drag and non-dimensionalize
    for wing in geometry.wings.values():
        ref = wing.areas.reference
        if wing.symmetric:
            wing_lifts[wing.tag]         = np.atleast_2d(np.sum(dim_wing_lifts[:,i:(i+2)],axis=1)).T/ref
            wing_drags[wing.tag]         = np.atleast_2d(np.sum(dim_wing_drags[:,i:(i+2)],axis=1)).T/ref
            wing_induced_angle[wing.tag] = np.concatenate((alpha_i[i],alpha_i[i+1]),axis=1)
            i+=1
        else:
            wing_lifts[wing.tag]         = np.atleast_2d(dim_wing_lifts[:,i]).T/ref
            wing_drags[wing.tag]         = np.atleast_2d(dim_wing_drags[:,i]).T/ref
            wing_induced_angle[wing.tag] = alpha_i[i]
        i+=1

    return total_lift_coeff, total_induced_drag_coeff, wing_lifts, wing_drags, cl_y, cdi_y, wing_induced_angle, CPi