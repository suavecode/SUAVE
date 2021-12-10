## @ingroup Analyses-Aerodynamics
# Lifting_Line.py
# 
# Created:  Aug 2017, E. Botero
#           Apr 2020, M. Clarke
#           Jun 2021, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Data, Units
from SUAVE.Methods.Aerodynamics.Lifting_Line import lifting_line as LL
from .Aerodynamics import Aerodynamics

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------
## @ingroup Analyses-Aerodynamics
class Lifting_Line(Aerodynamics):
    """This builds a surrogate and computes lift using a basic lifting line.

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
        self.tag = 'Lifting_Line'

        self.geometry = Data()
        self.settings = Data()

        # correction factors
        self.settings.fuselage_lift_correction           = 1.14
        self.settings.trim_drag_correction_factor        = 1.02
        self.settings.wing_parasite_drag_form_factor     = 1.1
        self.settings.fuselage_parasite_drag_form_factor = 2.3

        # vortex lattice configurations
        self.settings.number_of_stations  = 100
        
        # conditions table, used for surrogate model training
        self.training = Data()        
        self.training.angle_of_attack  = np.array([-10.,-5.,0.,5.,10.]) * Units.deg
        self.training.lift_coefficient = None
        self.training.drag_coefficient = None
        
        # surrogoate models
        self.surrogates = Data()
        self.surrogates.lift_coefficient = None
        self.surrogates.drag_coefficient = None
 
        
    def initialize(self,use_surrogate,n_sw,n_cw ,propeller_wake_model,use_bemt_wake_model,ito,wdt,nwts,mf,mn ,dcs):
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
        settings = self.settings
        
        if n_sw is not None:
            settings.number_of_stations  = n_sw
            
        # sample training data
        self.sample_training()
                    
        # build surrogate
        self.build_surrogate()


    def evaluate(self,state,settings,geometry):
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
          inviscid_wings[wings.*.tag]        [-] CL (wing specific)
          inviscid_wings.total               [-] CL
        conditions.aerodynamics.drag_breakdown.induced
          inviscid_wings[wings.*.tag]        [-] CDi (wing specific)
          total                              [-] CDi
          inviscid                           [-] CDi

        conditions.aerodynamics.
          inviscid_wings_lift                [-] CL

        Properties Used:
        self.surrogates.
          lift_coefficient                   [-] CL
          wing_lift_coefficient[wings.*.tag] [-] CL (wing specific)
          drag_coefficient                   [-] CDi
          wing_drag_coefficient[wings.*.tag] [-] CDi (wing specific)
        """          
        # unpack

        surrogates = self.surrogates        
        conditions = state.conditions
        
        # unpack        
        q    = conditions.freestream.dynamic_pressure
        AoA  = conditions.aerodynamics.angle_of_attack
        Sref = geometry.reference_area
        
        wings_lift_model = surrogates.lift_coefficient
        wings_drag_model = surrogates.drag_coefficient
        
        # inviscid lift of wings only
        inviscid_wings_lift                                                = Data()
        inviscid_wings_drag                                                = Data()
        inviscid_wings_lift.total                                          = wings_lift_model(AoA)
        inviscid_wings_drag.total                                          = wings_drag_model(AoA)        
        conditions.aerodynamics.lift_breakdown.inviscid_wings              = Data()
        conditions.aerodynamics.lift_breakdown.compressible_wings          = Data()
        conditions.aerodynamics.drag_breakdown.induced                     = Data()
        conditions.aerodynamics.drag_breakdown.induced.inviscid_wings      = Data()
        conditions.aerodynamics.lift_breakdown.inviscid_wings.total        = inviscid_wings_lift.total
        conditions.aerodynamics.lift_coefficient                           = inviscid_wings_lift.total
        conditions.aerodynamics.drag_breakdown.induced.total               = inviscid_wings_drag.total
        conditions.aerodynamics.drag_breakdown.induced.inviscid            = inviscid_wings_drag.total
        conditions.aerodynamics.drag_coefficient                           = inviscid_wings_drag.total        
        
        # store model for lift coefficients of each wing     
        for wing in geometry.wings.keys():
            wings_lift_model                                                         = surrogates.wing_lift_coefficients[wing] 
            wings_drag_model                                                         = surrogates.wing_drag_coefficients[wing]
            inviscid_wings_lift[wing]                                                = wings_lift_model(AoA)
            inviscid_wings_drag[wing]                                                = wings_drag_model(AoA)
            conditions.aerodynamics.lift_breakdown.inviscid_wings[wing]              = inviscid_wings_lift[wing] 
            conditions.aerodynamics.lift_breakdown.compressible_wings[wing]          = inviscid_wings_lift[wing]
            conditions.aerodynamics.drag_breakdown.induced.inviscid_wings[wing]      = inviscid_wings_drag[wing] 

        return inviscid_wings_lift , inviscid_wings_drag


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
          wing_lift_coefficients      [-] (wing specific)

        Properties Used:
        self.geometry.wings.*.tag
        self.settings                 (passed to calculate vortex lattice)
        self.training.angle_of_attack [radians]
        """        
        # unpack
        geometry = self.geometry
        settings = self.settings
        training = self.training
        
        AoA = training.angle_of_attack
        CL  = np.zeros_like(AoA)
        CDi = np.zeros_like(AoA)
        
        wing_CLs = Data.fromkeys(geometry.wings.keys(), np.zeros_like(AoA))
        wing_CDis = Data.fromkeys(geometry.wings.keys(), np.zeros_like(AoA)) 

        # condition input, local, do not keep
        konditions              = Data()
        konditions.aerodynamics = Data()

        # calculate aerodynamics for table
        for i,_ in enumerate(AoA):
            
            # overriding conditions, thus the name mangling
            konditions.aerodynamics.angle_of_attack = AoA[i]
            
            # these functions are inherited from Aerodynamics() or overridden
            CL[i], wing_lifts , CDi[i], wing_drags  = calculate_lift_lifting_line(konditions, settings, geometry)
            for wing in geometry.wings.values():
                wing_CLs[wing.tag][i] = wing_lifts[wing.tag]
                wing_CDis[wing.tag][i] = wing_drags[wing.tag]

        # store training data
        training.lift_coefficient       = CL
        training.wing_lift_coefficients = wing_CLs
        training.drag_coefficient       = CDi
        training.wing_drag_coefficients = wing_CDis        

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
          wing_lift_coefficients <np.poly1d> (multiple surrogates)

        Properties Used:
        self.
          training.
            angle_of_attack        [radians]
            lift_coefficient       [-]
            wing_lift_coefficients [-] (wing specific)
            drag_coefficient       [-]
            wing_drag_coefficients [-] (wing specific)
        """        
        # unpack data
        training      = self.training
        AoA_data      = training.angle_of_attack
        CL_data       = training.lift_coefficient
        wing_CL_data  = training.wing_lift_coefficients
        CDi_data      = training.drag_coefficient
        wing_CDi_data = training.wing_drag_coefficients        

        # pack for surrogate model
        X_data = np.array([AoA_data]).T
        X_data = np.reshape(X_data,-1)
        
        # learn the model
        cl_surrogate  = np.poly1d(np.polyfit(X_data, CL_data  ,1))
        cdi_surrogate = np.poly1d(np.polyfit(X_data, CDi_data ,2))
        
        wing_cl_surrogates = Data()
        wing_cdi_surrogates = Data()
        
        for wing in wing_CL_data.keys():
            wing_cl_surrogates[wing]  = np.poly1d(np.polyfit(X_data, wing_CL_data[wing] ,1))
            wing_cdi_surrogates[wing] = np.poly1d(np.polyfit(X_data, wing_CL_data[wing] ,2))

        self.surrogates.lift_coefficient       = cl_surrogate
        self.surrogates.drag_coefficient       = cdi_surrogate
        self.surrogates.wing_lift_coefficients = wing_cl_surrogates
        self.surrogates.wing_drag_coefficients = wing_cdi_surrogates

        return



# ----------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------


def calculate_lift_lifting_line(conditions,settings,geometry):
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

    # iterate over wings
    total_lift_coeff = 0.0
    total_drag_coeff = 0.0
    wing_lifts = Data()
    wing_drags = Data()
    for wing in geometry.wings.values():

        [wing_lift_coeff,wing_drag_coeff] = LL(conditions,settings,wing)
        total_lift_coeff += wing_lift_coeff * wing.areas.reference / vehicle_reference_area
        total_drag_coeff += wing_drag_coeff * wing.areas.reference / vehicle_reference_area
        wing_lifts[wing.tag] = wing_lift_coeff
        wing_drags[wing.tag] = wing_drag_coeff

    return total_lift_coeff, wing_lifts , total_drag_coeff , wing_drags
