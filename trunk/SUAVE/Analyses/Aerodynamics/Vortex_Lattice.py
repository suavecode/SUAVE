## @ingroup Analyses-Aerodynamics
# Vortex_Lattice.py
#
# Created:  May 2019, E. Botero
# Modified:    


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
import SUAVE

from SUAVE.Core import Data
from SUAVE.Core import Units

from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.VLM import VLM
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.weissinger_VLM import weissinger_VLM

# local imports
from .Aerodynamics import Aerodynamics
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_vortex_distribution import compute_vortex_distribution
from SUAVE.Plots import plot_vehicle_vlm_panelization 

# package imports
import numpy as np
import sklearn
from sklearn import gaussian_process
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
        self.tag = 'Vortex_Lattice'

        self.geometry = Data()
        self.settings = Data()

        # vortex lattice configurations
        self.settings.number_panels_spanwise   = 16
        self.settings.number_panels_chordwise  = 4
        self.settings.use_surrogate            = True
        self.settings.use_weissinger           = True
        self.settings.plot_vortex_distribution = False
        self.settings.plot_vehicle             = False
        self.settings.vortex_distribution      = Data()
        self.settings.call_function            = None

        
        # conditions table, used for surrogate model training
        self.training                          = Data()        
        self.training.angle_of_attack          = np.array([[-5.,-3.5,-2., 0.0, 2.0, 3.5, 5.0 , 8.0, 10., 12.]]).T * Units.deg
        self.training.Mach                     = np.array([[0.0,0.1 ,0.2, 0.3, 0.4, 0.5, 0.6 , 0.7, 0.8 ,0.9]]).T 
        self.training.lift_coefficient         = None
        self.training.wing_lift_coefficient    = None
        self.training.drag_coefficient         = None
        self.training.wing_drag_coefficient    = None
        
        # surrogoate models
        self.surrogates                        = Data()
        self.surrogates.lift_coefficient       = None  
        self.surrogates.wing_lift_coefficient  = None
        self.surrogates.drag_coefficient       = None
        self.surrogates.wing_drag_coefficient  = None
        
        self.evaluate                          = None
        
    def initialize(self):
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
        
        # Figure out if we are doing a full VLM or a Weissinger
        if   settings.use_weissinger == True:
            
            # Set the call function
            settings.call_function = calculate_weissinger
            
        elif settings.use_weissinger == False:
            
            # Set the call function
            settings.call_function = calculate_VLM
            
            # generate vortex distribution
            VD = compute_vortex_distribution(geometry,settings)      
            
            # Pack
            settings.vortex_distribution = VD
        
        # Plot vortex discretization of vehicle
        if settings.plot_vortex_distribution == True:
            plot_vehicle_vlm_panelization(VD)        
                
        # If we are using the surrogate
        if self.settings.use_surrogate == True:
            
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
        no changes to initial geometry or settings
        
        Source:
        N/A
        
        Inputs:
        state.conditions.
            angle_of_attack                       [radians]
        
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
        CL_surrogate        = surrogates.lift_coefficient
        CDi_surrogate       = surrogates.drag_coefficient
        wing_CL_surrogates  = surrogates.wing_lift_coefficient
        wing_CDi_surrogates = surrogates.wing_dragw_coefficient
                
        data_len                 = len(AoA)
        inviscid_lift            = np.zeros([data_len,1]) 
        inviscid_drag            = np.zeros([data_len,1])        
        
        for ii,_ in enumerate(AoA):
            inviscid_lift[ii]       = CL_surrogate.predict([np.array([AoA[ii][0],Mach[ii][0]])])  
            inviscid_drag[ii]       = CDi_surrogate.predict([np.array([AoA[ii][0],Mach[ii][0]])])            
            
            for wing in geometry.wings.keys():
                inviscid_wing_lifts = wing_CL_surrogates[wing].predict([np.array([AoA[ii][0],Mach[ii][0]])])  
                inviscid_wing_drags = wing_CDi_surrogates[wing].predict([np.array([AoA[ii][0],Mach[ii][0]])])
                conditions.aerodynamics.lift_breakdown.inviscid_wings_lift[wing]          = inviscid_wing_lifts
                conditions.aerodynamics.lift_breakdown.compressible_wings[wing]           = inviscid_wing_lifts      
                conditions.aerodynamics.drag_breakdown.induced.inviscid_wings_drag[wing]  = inviscid_wing_drags
         
        # Lift    
        conditions.aerodynamics.lift_coefficient                             = inviscid_lift*1.0
        conditions.aerodynamics.lift_breakdown.total                         = inviscid_lift*1.0
        
        # Drag   
        conditions.aerodynamics.drag_breakdown.induced                       = Data()
        conditions.aerodynamics.drag_breakdown.induced.total                 = inviscid_drag*1.0   
     
        return inviscid_lift
    
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
        inviscid_lift, inviscid_drag, wing_lifts, wing_drags, wing_lift_distribution , wing_drag_distribution , pressure_coefficient = \
            settings.call_function(conditions,settings,geometry)
        
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
          drag_coefficient            [-] 
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
        Mach     = training.Mach   
        
        # Setup Konditions                      
        konditions                              = Data()
        konditions.aerodynamics                 = Data()
        konditions.freestream                   = Data()
        konditions.aerodynamics.angle_of_attack = AoA 
        konditions.freestream.mach_number       = Mach
        
        # Assign placeholders        
        CL             = np.zeros([len(AoA)*len(Mach),1])
        CDi            = np.zeros([len(AoA)*len(Mach),1])
        CL_w           = Data()
        CDi_w          = Data()
        for wing in geometry.wings.keys():
            CL_w[wing] = np.zeros([len(AoA)*len(Mach),1])
            CDi_w[wing]= np.zeros([len(AoA)*len(Mach),1])
        
        # Calculate aerodynamics for table
        table_size     = len(AoA)*len(Mach)
        xy             = np.zeros([table_size,2])          
        for i,_ in enumerate(Mach):
            for j,_ in enumerate(AoA):
                xy[i*len(Mach)+j,:] = np.array([AoA[j][0],Mach[i][0]])
        
        # Get the training data        
        count = 0
        for j,_ in enumerate(Mach):
            konditions.freestream.mach_number = Mach[j]*np.ones_like(AoA)                         
            total_lift, total_drag, wing_lifts, wing_drags , wing_lift_distribution , wing_drag_distribution , pressure_coefficient = \
                settings.call_function(konditions,settings,geometry)
           
            # store training data
            CL[count*len(Mach):(count+1)*len(Mach),0]                = total_lift[:,0]
            CDi[count*len(Mach):(count+1)*len(Mach),0]               = total_drag[:,0]           
            for wing in geometry.wings.keys():
                CL_w[wing][count*len(Mach):(count+1)*len(Mach),0]    = wing_lifts[wing][:,0]
                CDi_w[wing][count*len(Mach):(count+1)*len(Mach),0]   = wing_drags[wing][:,0]                
            
            count += 1 
            
        # surrogate not run on sectional coefficients and pressure coefficients
        # Store training data
        training.grid_points              = xy
        training.lift_coefficient         = CL
        training.wing_lift_coefficient    = CL_w
        training.drag_coefficient         = CDi
        training.wing_drag_coefficient    = CDi_w
        
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
          lift_coefficient            <np.poly1d>
          wing_lift_coefficient       <np.poly1d> (multiple surrogates)
          drag_coefficient            <np.poly2d>
          wing_drag_coefficient       <np.poly2d> (multiple surrogates)
        Properties Used:
        self.training.
          lift_coefficient            [-] 
          wing_lift_coefficient       [-] (wing specific)
          drag_coefficient            [-] 
          wing_drag_coefficient       [-] (wing specific)
        """           

        # unpack data
        surrogates = self.surrogates
        training   = self.training
        geometry   = self.geometry
        AoA_data   = training.angle_of_attack
        mach_data  = training.Mach        
        CL_data    = training.lift_coefficient      
        CDi_data   = training.wing_lift_coefficient 
        CL_w_data  = training.drag_coefficient      
        CDi_w_data = training.wing_drag_coefficient 
        xy         = training.grid_points
        
        # Gaussian Process 
        regr_cl                    = gaussian_process.GaussianProcessRegressor()
        regr_cdi                   = gaussian_process.GaussianProcessRegressor()
        CL_surrogate               = regr_cl.fit(xy, CL_data)
        CDi_surrogate              = regr_cdi.fit(xy, CDi_data)        
        CL_w_surrogates            = Data() 
        CDi_w_surrogates           = Data()         
        for wing in geometry.wings.keys():
            regr_cl_w              = gaussian_process.GaussianProcessRegressor()
            regr_cdi_w             = gaussian_process.GaussianProcessRegressor()          
            CL_w_surrogates[wing]  = regr_cl_w.fit(xy, CL_w_data[wing])   
            CDi_w_surrogates[wing] = regr_cdi_w.fit(xy, CDi_w_data[wing])
   
        # Pack the outputs
        surrogates.lift_coefficient     = CL_surrogate
        surrogates.wing_lift_coefficient= CL_w_surrogates
        surrogates.drag_coefficient     = CDi_surrogate
        surrogates.wing_drag_coefficient= CDi_w_surrogates
        
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

    # unpack
    vehicle_reference_area = geometry.reference_area

    # iterate over wings
    wing_lifts = Data()
    wing_drags = Data()
    
    total_lift_coeff,total_induced_drag_coeff, CM, CL_wing, CDi_wing, cl_y , cdi_y , CPi = VLM(conditions,settings,geometry)

    ii = 0
    for wing in geometry.wings.values():
        wing_lifts[wing.tag] = 1*(np.atleast_2d(CL_wing[:,ii]).T)
        wing_drags[wing.tag] = 1*(np.atleast_2d(CDi_wing[:,ii]).T)
        ii+=1
        if wing.symmetric:
            ii+=1

    return total_lift_coeff, total_induced_drag_coeff, wing_lifts, wing_drags , cl_y , cdi_y , CPi



def calculate_weissinger(conditions,settings,geometry):
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
    
    Properties Used:
    
    """            

    # unpack
    vehicle_reference_area   = geometry.reference_area
    propulsors               = geometry.propulsors
    
    # iterate over wings
    wing_lifts               = Data()
    wing_drags               = Data()
    wing_lift_distribution   = Data()
    wing_drag_distribution   = Data()    
    
    total_lift_coeff         = 0.
    total_induced_drag_coeff = 0.

    for wing in geometry.wings.values():
        wing_lift_coeff,wing_drag_coeff, cl_y, cdi_y  =  weissinger_VLM(conditions,settings,wing,propulsors)
        total_lift_coeff                         += wing_lift_coeff * wing.areas.reference / vehicle_reference_area
        total_induced_drag_coeff                 += wing_drag_coeff * wing.areas.reference / vehicle_reference_area
        wing_lifts[wing.tag]                      = wing_lift_coeff
        wing_drags[wing.tag]                      = wing_drag_coeff
        wing_lift_distribution[wing.tag]          = cl_y
        wing_drag_distribution[wing.tag]          = cdi_y  
    
    return total_lift_coeff, total_induced_drag_coeff, wing_lifts, wing_drags, wing_lift_distribution , wing_drag_distribution, 0