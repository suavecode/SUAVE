""" Propulsor.py: parent class for propulsion systems """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Structure import Data, Data_Exception, Data_Warning
from SUAVE.Components import Component, Physical_Component, Lofted_Body
from SUAVE.Attributes.Gases import Air

# ----------------------------------------------------------------------
#  Propulsor
# ----------------------------------------------------------------------

class Propulsor(Physical_Component):

    """ A component that makes go-ification """

    def __defaults__(self):
        self.tag = 'Propulsor'
        self.breathing = True
        self.gas = Air()

    def get_area(self):

        # get or determine intake area
        try: 
            self.A
        except AttributeError:                  # no area given
            try: 
                self.D
            except AttributeError:              # no diameter given
                print "Error in Propulsor: no diameter or area given."
                return []
            else:
                A = np.pi*(self.D/2)**2
        else:                                   # area given
            if self.A > 0.0:
                A = self.A
            else:
                try: 
                    self.D
                except AttributeError:          # no diameter given
                    print "Error in Propulsor: no diameter given and area <= 0."
                    return []
                else:
                    A = np.pi*(self.D/2)**2

        return A

class Container(Physical_Component.Container):
    """ Contains many SUAVE.Components.Propulsor()
    
        Search Methods
            import SUAVE.Components.Propulsors
            example: find_instances(Propulsors.Motor)    > return all Motors
            example: find_instances(Propulsors.Turbojet) > return all Turbojets
    """
    
    def __call__(self,conditions,numerics):
        
        segment=Data()
        segment.q  = conditions.freestream.dynamic_pressure[:,0]
        segment.g0 = conditions.freestream.gravity[:,0]
        segment.V  = conditions.freestream.velocity[:,0]
        segment.M  = conditions.freestream.mach_number[:,0]
        segment.T  = conditions.freestream.temperature[:,0]
        segment.p  = conditions.freestream.pressure[:,0]
        
        eta        = conditions.propulsion.throttle[:,0]
        
        F    = np.zeros_like(eta)
        mdot = np.zeros_like(eta)
        P    = np.zeros_like(eta)
        
        for propulsor in self.values():
            CF, Isp, etaPe = propulsor(eta,segment)

            # get or determine intake area
            A = propulsor.get_area()

            # compute data
            F += CF*segment.q*A                             # N
            
            # propellant-based
            if np.isscalar(Isp):
                if Isp != 0.0:
                    mdot += F/(Isp*segment.g0)              # kg/s
                    
            else:
                mask = (Isp != 0.0)
                mdot[mask] += F[mask]/(Isp[mask]*segment.g0)   # kg/s
                
            # electric-based
            if np.isscalar(etaPe):
                if etaPe != 0.0:
                    P += F*segment.V/etaPe                  # W
                    
                   #Account for mass gain of Li-air battery
                    try:
                        self.battery
                    except AttributeError:
                        
                        if propulsor.battery.type=='Li-Air': 
                            
                            for i in range(len(P)):
                                if propulsor.battery.MaxPower>P[i]:
                                    [Ploss,Mdot]=propulsor.battery(P[i],.01 )       #choose small dt here (has not been solved for yet); its enough to find mass rate gain of battery
                                else:
                                    [Ploss,Mdot]=propulsor.battery(propulsor.battery.MaxPower,.01 )
                                mdot[i]+=Mdot.real
                                
                                
                                
                                
                   
            else:
                mask = (etaPe != 0.0)
                P += F[mask]*segment.V[mask]/etaPe[mask]    # W
            #print mdot   
            
        return F, mdot, P

    def power_flow(self,eta,segment):

        P_fuel = np.zeros_like(eta) 
        P_e = np.zeros_like(eta)

        for propulsor in self.values():
            CF, Isp, etaPe = propulsor(eta,segment)

            # get basic data
            A = propulsor.get_area()
            F = CF*segment.q*A                                      # N

            # propellant-based
            if np.isscalar(Isp):
                if Isp != 0.0:
                    mdot = F/(Isp*segment.g0)                       # kg/s
                    
                else:
                    mdot = 0.0
                    propulsor.propellant.specific_energy=0.0
                    
            else:
                mask = (Isp != 0.0)
                mdot = np.zeros_like(F)
                mdot[mask] = F[mask]/(Isp[mask]*segment.g0)         # kg/s
                P_fuel=np.zeros_like(F)
                P_fuel+=mdot[mask]
            P_fuel += mdot*propulsor.propellant.specific_energy      # W

            # electric-based
            if np.isscalar(etaPe):
                if etaPe != 0.0:
                    P_e += F*segment.V/etaPe
            else:
                mask = (etaPe != 0.0)
                P_e = np.zeros_like(F)
                P_e[mask] += F[mask]*segment.V[mask]/etaPe[mask]        # W

        return P_fuel, P_e

# ----------------------------------------------------------------------
#  Handle Linking
# ----------------------------------------------------------------------

Propulsor.Container = Container
    



# --------------- GRAVE TART ---------------------
    #def get_DuctedFans(self):

        #from Ducted_Fan import Ducted_Fan
        #return self.find_instances(self,Ducted_Fan,index)

    #def get_InternalCombustions(self,index=None):

        #from Internal_Combustion import Internal_Combustion
        #return self.find_instances(self,Internal_Combustion,index)

    #def get_TurboFans(self,index=None):
        #from Turbo_Fan import Turbo_Fan
        #return self.find_instances(self,Turbo_Fan,index)

    #def get_Transmissions(self,index=None):
        #from Transmission import Transmission
        #return self.find_instances(self,Transmission,index)

    #def get_ElectricMotors(self,index=None):
        #from Electric_Motor import Electric_Motor
        #return self.find_instances(self,Electric_Motor,index)
        
    #def get_MotorFCs(self,index=None):
        #from Motor_FC import Motor_FC
        #return self.find_instances(self,Motor_FC,index)
    
    #def get_MotorBats(self,index=None):
        #from Motor_Bat import Motor_Bat
        #return self.find_instances(self,Motor_Bat,index)

    #def get_Rotors(self,index=None):
        #from Rotor import Rotor
        #return self.find_instances(self,Rotor,index)    