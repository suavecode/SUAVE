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
from SUAVE.Plots import plot_vehicle_geometry
# package imports
import numpy as np

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
        self.training = Data()        
        self.training.angle_of_attack       = np.array([[-10.,-8., -5.,-2. ,0.,2.,5.,8.,10.]]).T * Units.deg
        self.training.lift_coefficient      = None
        self.training.wing_lift_coefficient = None
        self.training.drag_coefficient      = None
        self.training.wing_drag_coefficient = None
        
        # surrogoate models
        self.surrogates = Data()
        self.surrogates.lift_coefficient = None        
        
        self.evaluate = None
        
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
        
        # Plot vortex discretization of vehicle
        if settings.plot_vehicle == True:
            plot_vehicle_geometry(VD)    
                
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
            angle_of_attack                    [radians]
        Outputs:
        conditions.aerodynamics.lift_breakdown.
          inviscid_wings_lift[wings.*.tag]   [-] CL (wing specific)
          inviscid_wings_lift.total          [-] CL
        conditions.aerodynamics.
          lift_coefficient_wing              [-] CL (wing specific)
        inviscid_wings_lift                  [-] CL
        Properties Used:
        self.surrogates.
          lift_coefficient                   [-] CL
          wing_lift_coefficient[wings.*.tag] [-] CL (wing specific)
        """          
        
        # unpack        
        conditions = state.conditions
        settings   = self.settings
        geometry   = self.geometry
        surrogates = self.surrogates
        AoA        = conditions.aerodynamics.angle_of_attack
           
        # Unapck the surrogates
        CL_surrogate        = surrogates.lift_coefficient
        CDi_surrogate       = surrogates.drag_coefficient
        wing_CL_surrogates  = surrogates.wing_lifts 
        wing_CDi_surrogates = surrogates.wing_drags
        
        # Evaluate the surrogate
        inviscid_lift = CL_surrogate(AoA)
        inviscid_drag = CDi_surrogate(AoA)
        
        # Pull out the individual lifts
        wing_lifts = Data()
        wing_drags = Data()
        
        for key in geometry.wings.keys():
            wing_lifts[key] = wing_CL_surrogates[key](AoA)
            wing_drags[key] = wing_CDi_surrogates[key](AoA)
        
        # Lift    
        conditions.aerodynamics.lift_coefficient                             = inviscid_lift
        conditions.aerodynamics.lift_breakdown.total                         = inviscid_lift        
        conditions.aerodynamics.lift_breakdown.compressible_wings            = wing_lifts
        conditions.aerodynamics.lift_breakdown.inviscid_wings_lift           = wing_lifts
        
        # Drag   
        conditions.aerodynamics.drag_breakdown.induced                       = Data()
        conditions.aerodynamics.drag_breakdown.induced.total                 = inviscid_drag
        conditions.aerodynamics.drag_breakdown.induced.inviscid_wings_drag   = wing_drags       
                
        return inviscid_lift
    
    def evaluate_no_surrogate(self,state,settings,geometry):
        """Evaluates lift and drag directly using VLM
        
        Assumptions:
        no changes to initial geometry or settings
        Source:
        N/A
        Inputs:
        state.conditions.
          angle_of_attack                    [radians]
        Outputs:
        conditions.aerodynamics.lift_breakdown.
          inviscid_wings_lift[wings.*.tag]   [-] CL (wing specific)
          inviscid_wings_lift.total          [-] CL
        
        conditions.aerodynamics.
          drag_breakdown.induced.total       [-] CDi (wing specific)
          lift_coefficient_wing              [-] CL (wing specific)
          drag_coefficient_wing              [-] CDi (wing specific)
        inviscid_wings_lift                  [-] CL
        Properties Used:
        self.surrogates.
          lift_coefficient                   [-] CL
          wing_lift_coefficient[wings.*.tag] [-] CL (wing specific)
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
          wing_lifts                  [-] (wing specific)
          drag_coefficient            [-] 
          wing_drags                  [-] (wing specific)
        Properties Used:
        self.geometry.wings.*.tag
        self.settings                 (passed to calculate vortex lattice)
        self.training.angle_of_attack [radians]
        """  
        
        # unpack
        geometry = self.geometry
        settings = self.settings
        training = self.training
        
        # Setup Konditions
        konditions              = Data()
        konditions.aerodynamics = Data()
        konditions.aerodynamics.angle_of_attack = training.angle_of_attack
        
        # Get the training data        
        total_lift, total_drag, wing_lifts, wing_drags , wing_lift_distribution , wing_drag_distribution , pressure_coefficient = \
            settings.call_function(konditions,settings,geometry)
        
        # surrogate not run on sectional coefficients and pressure coefficients
        # Store training data
        training.lift_coefficient = total_lift
        training.drag_coefficient = total_drag
        training.wing_lifts       = wing_lifts
        training.wing_drags       = wing_drags
        
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
          lift_coefficient       <np.poly1d>
          wing_inviscid_lift <np.poly1d> (multiple surrogates)
          drag_coefficient       <np.poly2d>
          wing_inviscid_drag <np.poly2d> (multiple surrogates)
        Properties Used:
        self.training.
          lift_coefficient            [-] 
          wing_lifts                  [-] (wing specific)
          drag_coefficient            [-] 
          wing_drags                  [-] (wing specific)
        """           

        # unpack data
        training      = self.training
        AoA_data      = training.angle_of_attack
        CL_data       = training.lift_coefficient
        CDi_data      =  training.drag_coefficient
        wing_CL_data  = training.wing_lifts
        wing_CDi_data = training.wing_drags    
        
        # learn the models
        CL_surrogate  = np.poly1d(np.polyfit(AoA_data.T[0], CL_data.T[0], 1))
        CDi_surrogate = np.poly1d(np.polyfit(AoA_data.T[0], CDi_data.T[0], 2))
        
        wing_CL_surrogates = Data()
        wing_CDi_surrogates = Data()
        
        for wing in wing_CL_data.keys():
            wing_CL_surrogates[wing] = np.poly1d(np.polyfit(AoA_data.T[0], wing_CL_data[wing].T[0], 1))   
            wing_CDi_surrogates[wing] = np.poly1d(np.polyfit(AoA_data.T[0], wing_CDi_data[wing].T[0], 2))   
        
        # Pack the outputs
        self.surrogates.lift_coefficient = CL_surrogate
        self.surrogates.drag_coefficient = CDi_surrogate
        self.surrogates.wing_lifts       = wing_CL_surrogates
        self.surrogates.wing_drags       = wing_CDi_surrogates
        
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
    total_lift_coeff          [array]
    total_induced_drag_coeff  [array]
    wing_lifts                [Data]
    wing_drags                [Data]
    Properties Used:
    
    """            

    # unpack
    vehicle_reference_area = geometry.reference_area

    # iterate over wings
    wing_lifts = Data()
    wing_drags = Data()
    
    total_lift_coeff,total_induced_drag_coeff, CM, CL_wing, CDi_wing, cl_sec , cdi_sec , CPi = VLM(conditions,settings,geometry)

    ii = 0
    for wing in geometry.wings.values():
        wing_lifts[wing.tag] = 1*(np.atleast_2d(CL_wing[:,ii]).T)
        wing_drags[wing.tag] = 1*(np.atleast_2d(CDi_wing[:,ii]).T)
        ii+=1
        if wing.symmetric:
            wing_lifts[wing.tag] += 1*(np.atleast_2d(CL_wing[:,ii]).T)
            wing_drags[wing.tag] += 1*(np.atleast_2d(CDi_wing[:,ii]).T)
            ii+=1

    return total_lift_coeff, total_induced_drag_coeff, wing_lifts, wing_drags , cl_sec , cdi_sec , CPi



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
    vehicle_reference_area = geometry.reference_area
    propulsors             = geometry.propulsors
    
    # iterate over wings
    wing_lifts = Data()
    wing_drags = Data()
    wing_lift_distribution = Data()
    wing_drag_distribution = Data()    
    
    total_lift_coeff         = 0.
    total_induced_drag_coeff = 0.

    for wing in geometry.wings.values():
        wing_lift_coeff,wing_drag_coeff, cl, cdi  =  weissinger_VLM(conditions,settings,wing,propulsors)
        total_lift_coeff += wing_lift_coeff * wing.areas.reference / vehicle_reference_area
        total_induced_drag_coeff += wing_drag_coeff * wing.areas.reference / vehicle_reference_area
        wing_lifts[wing.tag] = wing_lift_coeff
        wing_drags[wing.tag] = wing_drag_coeff
        wing_lift_distribution[wing.tag] = cl
        wing_drag_distribution[wing.tag] = cdi   
    return total_lift_coeff, total_induced_drag_coeff, wing_lifts, wing_drags, wing_lift_distribution , wing_drag_distribution, 0