""" Segment.py: parent class for Propulsor Segments """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Components import Component
from SUAVE.Structure import Data, Data_Exception, Data_Warning
from SUAVE.Attributes.Gases import Air
from SUAVE.Attributes.Propellants import Propellant

# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

class Segment(Component):

    """ A Segment of a Propulsor """

    def __defaults__(self):

        self.tag = 'Segment'
        self.active = True
        self.i = 0; self.f = 0

        # property ratios       
        self.p_ratio = 1.0
        self.T_ratio = 1.0
        self.pt_ratio = 1.0
        self.Tt_ratio = 1.0

        # efficiencies 
        self.eta_polytropic = None
        self.eta = 1.0

        # interacts with 1D flow?
        self.flow = True


class Container(Component.Container):
    pass

#class Container(Physical_Component.Container):
#    """ Contains many SUAVE.Components.Propulsor()
    
#        Get Methods 
#            get_TurboFans()
#            get_Transmissions()
#            get_ElectricMotors()
#            get_Rotors()
#            get_instance(ComponentType)
#    """
    
#    def get_DuctedFans(self,index=None):
#        from Ducted_Fan import Ducted_Fan
#        return self.find_instances(self,Ducted_Fan,index)

#    def get_InternalCombustions(self,index=None):
#        from Internal_Combustion import Ducted_Fan
#        return self.find_instances(self,Internal_Combustion,index)

#    def get_TurboFans(self,index=None):
#        from Turbo_Fan import Turbo_Fan
#        return self.find_instances(self,Turbo_Fan,index)

#    def get_Transmissions(self,index=None):
#        from Transmission import Transmission
#        return self.find_instances(self,Transmission,index)

#    def get_ElectricMotors(self,index=None):
#        from Electric_Motor import Electric_Motor
#        return self.find_instances(self,Electric_Motor,index)
        
#    def get_MotorFCs(self,index=None):
#        from Motor_FC import Motor_FC
#        return self.find_instances(self,Motor_FC,index)
    
#    def get_MotorBats(self,index=None):
#        from Motor_Bat import Motor_Bat
#        return self.find_instances(self,Motor_Bat,index)

#    def get_Rotors(self,index=None):
#        from Rotor import Rotor
#        return self.find_instances(self,Rotor,index)    

#    def __call__(self,eta,segment):

#        F = np.zeros_like(eta)
#        mdot = np.zeros_like(eta)
#        P = np.zeros_like(eta)

#        for propulsor in self.values():
#            CF, Isp, etaPe = propulsor(eta,segment)

#            # get or determine intake area
#            A = propulsor.get_area()

#            # compute data
#            F += CF*segment.q*A                             # N

#            # propellant-based
#            if np.isscalar(Isp):
#                if Isp != 0.0:
#                    mdot += F/(Isp*segment.g0)              # kg/s
#            else:
#                mask = (Isp != 0.0)
#                mdot[mask] += F[mask]/(Isp[mask]*segment.g0)   # kg/s

#            # electric-based
#            if np.isscalar(etaPe):
#                if etaPe != 0.0:
#                    P += F*segment.V/etaPe                  # W
#            else:
#                mask = (etaPe != 0.0)
#                P += F[mask]*segment.V[mask]/etaPe[mask]    # W

#        return F, mdot, P

#    def power_flow(self,eta,segment):

#        P_fuel = np.zeros_like(eta) 
#        P_e = np.zeros_like(eta)

#        for propulsor in self.values():
#            CF, Isp, etaPe = propulsor(eta,segment)

#            # get basic data
#            A = propulsor.get_area()
#            F = CF*segment.q*A                                      # N

#            # propellant-based
#            if np.isscalar(Isp):
#                if Isp != 0.0:
#                    mdot = F/(Isp*segment.g0)                       # kg/s
                    
#                else:
#                    mdot = 0.0
#                    propulsor.propellant.specific_energy=0.0
                    
#            else:
#                mask = (Isp != 0.0)
#                mdot = np.zeros_like(F)
#                mdot[mask] = F[mask]/(Isp[mask]*segment.g0)         # kg/s
#                P_fuel=np.zeros_like(F)
#                P_fuel+=mdot[mask]
#            P_fuel += mdot*propulsor.propellant.specific_energy      # W

#            # electric-based
#            if np.isscalar(etaPe):
#                if etaPe != 0.0:
#                    P_e += F*segment.V/etaPe
#            else:
#                mask = (etaPe != 0.0)
#                P_e = np.zeros_like(F)
#                P_e[mask] += F[mask]*segment.V[mask]/etaPe[mask]        # W

#        return P_fuel, P_e
    

# ----------------------------------------------------------------------
#  Handle Linking
# ----------------------------------------------------------------------

Segment.Container = Container
    
