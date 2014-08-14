#solar_energy_network.py
# 
# Created:  Emilio Botero, Jun 2014
# Modified:  

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
import scipy as sp
import datetime
import time
from SUAVE.Attributes import Units

from SUAVE.Structure import (
Data, Container, Data_Exception, Data_Warning,
)

# ----------------------------------------------------------------------
#   Alternate Approach 
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
#  Energy Component Class
# ----------------------------------------------------------------------
from SUAVE.Components import Physical_Component

class Energy_Component(Physical_Component):
    def __defaults__(self):
        
        # function handles for input
        self.inputs  = Data()
        
        # function handles for output
        self.outputs = Data()
        
        return

# ----------------------------------------------------------------------
#  Solar Class
# ----------------------------------------------------------------------
class solar(Energy_Component):

    def solar_flux(self,conditions):  
        
        """ Computes the adjusted solar flux in watts per square meter.
              
              Inputs:
                  day - day of the year from Jan 1st
                  TUTC - time in seconds in UTC
                  longitude- in degrees
                  latitude - in degrees
                  altitude - in meters                  
                  bank angle - in radians
                  pitch attitude - in radians
                  heading angle - in degrees
                  
              Outputs:
                  sflux - adjusted solar flux
                  
              Assumptions:
                  Solar intensity =1367 W/m^2
                  Includes a diffuse component of 10% of the direct component
                  Altitudes are not excessive 
        """        
        
        #unpack
        timedate  = conditions.frames.planet.timedate
        latitude  = conditions.frames.planet.lat
        longitude = conditions.frames.planet.lon
        phip      = conditions.frames.body.inertial_rotations[:,0]
        thetap    = conditions.frames.body.inertial_rotations[:,1]
        psip      = conditions.frames.body.inertial_rotations[:,2]
        altitude  = conditions.freestream.altitude
        times     = conditions.frames.inertial.time
        
        #Figure out the date and time
        day       = timedate.tm_yday + np.floor_divide(times, 24.*60.*60.)
        TUTC      = timedate.tm_sec + 60.*timedate.tm_min+ 60.*60.*timedate.tm_hour + np.mod(times,24.*60.*60.)
        
        #Gamma is defined to be due south, so
        gamma = psip-np.pi
        
        #Solar intensity external to the Earths atmosphere
        Io = 1367.0
        
        #B
        B = (360./365.0)*(day-81.)*np.pi/180.0
        
        #Equation of Time
        EoT = 9.87*np.sin(2*B)-7.53*np.cos(B)-1.5*np.sin(B)
        
        #Time Correction factor
        TC = 4*(longitude*np.pi/180)+EoT
        
        #Local Solar Time
        LST = TUTC/3600.0+TC/60.0
        
        #Hour Angle   
        HRA = (15.0*(LST-12.0))*np.pi/180.0
        
        #Declination angle (rad)
        delta = -23.44*np.cos((360./365.)*(day+10.)*np.pi/180.)*np.pi/180.
        
        #Zenith angle (rad)
        psi = np.arccos(np.sin(delta)*np.sin(latitude*np.pi/180.0)+np.cos(delta)*np.cos(latitude*np.pi/180.0)*np.cos(HRA))
        
        #Solar Azimuth angle, Duffie/Beckman 1.6.6
        gammas = np.sign(HRA)*np.abs((np.cos(psi)*np.sin(latitude*np.pi/180.)-np.sin(delta))/(np.sin(psi)*np.cos(latitude*np.pi/180)))
        
        #Slope of the solar panel, Bower AIAA 2011-7072 EQN 15
        beta = np.arccos(np.cos(thetap)*np.cos(phip))
        
        #Angle of incidence, Duffie/Beckman 1.6.3
        theta = np.arccos(np.cos(psi)*np.cos(beta)+np.sin(psi)*np.sin(beta)*np.cos(gammas-gamma))
        
        flux = np.zeros_like(psi)

        for ii in range(len(psi[:,0])):
        
            if (psi[ii,0]>=-np.pi/2.)&(psi[ii,0]<96.70995*np.pi/180.)&(altitude[ii,0]<9000.):
                 
                #Using a homogeneous spherical model
                earthstuff = SUAVE.Attributes.Planets.Earth()
                Re = earthstuff.mean_radius
                 
                Yatm = 9. #The atmospheres thickness in km
                r = Re/Yatm
                c = altitude[ii,0]/9000. #Converted from m to km
                 
                AM = (((r+c)**2)*(np.cos(psi[ii,0])**2)+2.*r*(1.-c)-c**2 +1.)**(0.5)-(r+c)*np.cos(psi[ii,0])
                 
                Id = 1.1*Io*(0.7**(AM**0.678))
                
                #Horizontal Solar Flux on the panel
                Ih = Id*(np.cos(latitude*np.pi/180.)*np.cos(delta[ii,0])*np.cos(HRA[ii,0])+np.sin(latitude*np.pi/180.)*np.sin(delta[ii,0]))              
                 
                #Solar flux on the inclined panel, Duffie/Beckman 1.8.1
                I = Ih*np.cos(theta[ii,0])/np.cos(psi[ii,0])
                 
            elif (psi[ii,0]>=-np.pi/2.)&(psi[ii,0]<96.70995*np.pi/180.)&(altitude[ii,0]>=9000.):
                 
                Id = 1.1*Io
                
                #Horizontal Solar Flux on the panel
                Ih = Id*(np.cos(latitude*np.pi/180.)*np.cos(delta[ii,0])*np.cos(HRA[ii,0])+np.sin(latitude*np.pi/180.)*np.sin(delta[ii,0]))           
       
                #Solar flux on the inclined panel, Duffie/Beckman 1.8.1
                I = Ih*np.cos(theta[ii,0])/np.cos(psi[ii,0])
                 
            else:
                I = 0.
             
            #Adjusted Solar flux on a horizontal panel
            flux[ii,0] = max(0.,I)
        
        # store to outputs
        self.outputs.flux = flux      
        
        #print('Theta of the vehicle')
        #print(thetap)
        
        # return result for fun/convenience
        return flux

# ----------------------------------------------------------------------
#  Solar_Panel Class
# ----------------------------------------------------------------------
class solar_panel(Energy_Component):
    
    def __defaults__(self):
        self.A   = 0.0
        self.eff = 0.0
    
    def power(self):
        flux=self.inputs.flux
        
        p = flux*self.A*self.eff
        
        # store to outputs
        self.outputs.power = p
    
        return p
    
# ----------------------------------------------------------------------
#  Motor Class
# ----------------------------------------------------------------------
    
class Motor(Energy_Component):
    
    def __defaults__(self):
        
        self.Res = 0.0
        self.io  = 0.0
        self.kv  = 0.0
        self.propradius   = 0.0
        self.propCp  = 0.0
    
    def omega(self,conditions):
        """ The motor's rotation rate
            
            Inputs:
                Motor resistance - in ohms
                Motor zeros load current - in amps
                Motor Kv - in rad/s/volt
                Propeller radius - in meters
                Propeller Cp - power coefficient
                Freestream velocity - m/s
                Freestream dynamic pressure - kg/m/s^2
                
            Outputs:
                The motor's rotation rate
               
            Assumptions:
                Cp is not a function of rpm or RE
               
        """
        #Unpack
        V     = conditions.freestream.velocity[:,0]
        rho   = conditions.freestream.density[:,0]
        Res   = self.Res
        etaG  = self.etaG
        exp_i = self.exp_i
        io    = self.io + exp_i*(1-etaG)
        G     = self.G
        Kv    = self.kv/G
        R     = self.propradius
        Cp    = self.propCp
        v     = self.inputs.voltage[:,0]

        #Omega
        #This is solved by setting the torque of the motor equal to the torque of the prop
        #It assumes that the Cp is constant
        omega1 =   ((- 2.*Cp*np.pi*io*rho*Kv**3.*R**5.*Res**2. +
                     2.*Cp*np.pi*rho*v*Kv**3.*R**5.*Res + 1.)**(0.5) - 
                    1.)/(Cp*Kv**2.*np.pi*R**5.*Res*rho)
        
            #print('Omega1')
            #print(omega1)     
  
        # store to outputs
        self.outputs.omega = omega1
        
        #Q = ((v-omega1/Kv)/Res -io)/Kv
        #print('Motor Torque')
        #print(Q)
        
        #P = Q*omega1
        
        #print('Motor Power Output')
        #print P
        
        return omega1*G
    
    def current(self,conditions):
        """ The motor's current
            
            Inputs:
                Motor resistance - in ohms
                Motor Kv - in rad/s/volt
                
            Outputs:
                The motor's rpm, current, thrust, and power out
               
            Assumptions:
                Cp is invariant
               
        """    
        
        Kv   = self.kv/self.G
        Res  = self.Res
        v    = self.inputs.voltage[:,0]
        etaG = self.etaG
        G    = self.G
        io   = self.io + self.exp_i*(1-etaG)
        omeg = self.omega(conditions)/G
        
        i=(v-omeg/Kv)/Res

        # store to outputs
        self.outputs.current = i
        
        ##Recalculate motor torque
        #Q = (i-io)/Kv
        #print('Motor torque from Current')
        #print(Q)
        
        #print('Motor input Power')
        #print(i*v)
        
        #print('Pshaft')
        #pshaft= (i-io)*(v-i*Res)   
        #print(pshaft)
        
        print('Motor Efficiency')
        etam=(1-io/i)*(1-i*Res/v)
        print(etam)
        
        
        return i
    
# ----------------------------------------------------------------------
#  Propeller Class
# ----------------------------------------------------------------------    
 
class Propeller(Energy_Component):
    
    def __defaults__(self):
        
        self.Ct     = 0.0
        self.Cp     = 0.0
        self.radius = 0.0
    
    def spin(self,conditions):
        """ Analyzes a propeller given geometry and operating conditions
                 
                 Inputs:
                     hub radius
                     tip radius
                     rotation rate
                     freestream velocity
                     number of blades
                     number of stations
                     chord distribution
                     twist distribution
                     airfoil data
       
                 Outputs:
                     Power coefficient
                     Thrust coefficient
                     
                 Assumptions:
                     Based on Qprop Theory document
       
           """
           
        #Unpack    
        B     = self.Prop_attributes.B      # Number of Blades
        R     = self.Prop_attributes.R      # Tip Radius
        Rh    = self.Prop_attributes.Rh     # Hub Radius
        beta  = self.Prop_attributes.beta   # Twist
        c     = self.Prop_attributes.c      # Chord distribution
        omega = self.inputs.omega           # Rotation Rate in rad/s
        rho   = conditions.freestream.density
        mu    = conditions.freestream.viscosity
        V     = conditions.freestream.velocity[:,0]
        a     = conditions.freestream.speed_of_sound
        
        nu    = mu/rho
        tol   = 1e-8 # Convergence tolerance
           
        ######
        #Figure out how to enter airfoil data
        ######

        #Things that don't change with iteration
        N       = len(c) #Number of stations
        chi0    = Rh/R # Where the propeller blade actually starts
        chi     = np.linspace(chi0,1,N+1) # Vector of nondimensional radii
        chi     = chi[0:N]
        lamda   = V/(omega*R)           # Speed ratio
        r       = chi*R                 # Radial coordinate

        x       = r*np.dot(omega,1/V)             # Nondimensional distance
        n       = omega/(2.*np.pi)      # Cycles per second
        J       = V/(2.*R*n)    
    
        sigma   = np.multiply(B*c,1./(2.*np.pi*r))   
    
        #I make the assumption that externally-induced velocity at the disk is zero
        #This can be easily changed if needed in the future:
        ua = 0.0
        ut = 0.0
        
        omegar = np.outer(omega,r)
        Ua = np.outer((V + ua),np.ones_like(r))
        Ut = omegar - ut
        U  = np.sqrt(Ua**2. + Ut**2.)
        
        #Things that will change with iteration
    
        #Setup a Newton iteration
        psi =  np.ones_like(c)
        psiold = np.zeros_like(c)
        diff = np.ones_like(c)
        
        while (np.any(diff>tol)):
            print(psi)
            #Wa    = 0.5*Ua + 0.5*U*np.sin(psi)
            #Wt    = 0.5*Ut + 0.5*U*np.cos(psi)
            Wa    = Ua + 0.5*U*np.sin(psi)
            Wt    = Ut + 0.5*U*np.cos(psi)            
            va    = Wa - Ua
            vt    = Ut - Wt
            alpha = beta - np.arctan2(Wa,Wt)
            W     = np.sqrt(Wa**2. + Wt**2.)
            Re    = W*c*nu
            #Ma    = W/a #a is the speed of sound
            
            lamdaw = r*Wa/(R*Wt)
            f      = (B/2.)*(1.-r/R)/lamdaw
            F      = 2.*np.arccos(np.exp(-f))/np.pi
            Gamma  = vt*(4.*np.pi*r/B)*F*np.sqrt(1.+(4.*lamdaw*R/(np.pi*B*r))**2.)
            
            #Ok, from the airfoil data, given Re, Ma, alpha we need to find Cl
            Cl = 2.*np.pi*alpha
            
            Rsquiggly = Gamma - 0.5*W*c*Cl   
            
            #An analytical derivative for dR_dpsi, this is derived by taking a derivative of the above equations
            #This was solved symbolically in Matlab and exported
            dR_dpsi = (4.*U*r*np.arccos(np.exp(-(B*(Ut + U*np.cos(psi))*(R - r))/(2.*r*(Ua + U*np.sin(psi)))))*np.sin(psi)*((16.*(Ua +
                      U*np.sin(psi))**2.)/(B**2.*np.pi**2.*(Ut + U*np.cos(psi))**2.) + 1.)**(0.5))/B - (np.pi*U*(Ua*np.cos(psi) - 
                      Ut*np.sin(psi))*(beta - np.arctan((Ua + U*np.sin(psi))/(Ut + U*np.cos(psi)))))/(2.*((Ut + U*np.cos(psi))**2. +
                      (Ua + U*np.sin(psi))**2.)**(0.5)) + (np.pi*U*((Ut + U*np.cos(psi))**2. + (Ua + U*np.sin(psi))**2.)**(0.5)*(U +
                      Ut*np.cos(psi) + Ua*np.sin(psi)))/(2.*((Ua + U*np.sin(psi))**2./(Ut + U*np.cos(psi))**2. + 1.)*(Ut +
                      U*np.cos(psi))**2.) - (4.*U*np.exp(-(B*(Ut + U*np.cos(psi))*(R - r))/(2.*r*(Ua + U*np.sin(psi))))*((16.*(Ua +
                      U*np.sin(psi))**2.)/(B**2.*np.pi**2.*(Ut + U*np.cos(psi))**2.) + 1.)**(0.5)*(R - r)*(Ut/2. - (U*np.cos(psi))/2.)*(U + 
                      Ut*np.cos(psi) + Ua*np.sin(psi)))/((Ua + U*np.sin(psi))**2.*(1. - np.exp(-(B*(Ut + U*np.cos(psi))*(R - r))/(r*(Ua + 
                      U*np.sin(psi)))))**(0.5)) + (128.*U*r*np.arccos(np.exp(-(B*(Ut + U*np.cos(psi))*(R - r))/(2.*r*(Ua + U*np.sin(psi)))))*(Ua +
                      U*np.sin(psi))*(Ut/2. - (U*np.cos(psi))/2.)*(U + Ut*np.cos(psi) + Ua*np.sin(psi)))/(B**3.*np.pi**2.*(Ut +
                      U*np.cos(psi))**3.*((16.*(Ua + U*np.sin(psi))**2.)/(B**2.*np.pi**2.*(Ut + U*np.cos(psi))**2.) + 1.)**(0.5))
                      
            dpsi = -Rsquiggly/dR_dpsi
            
            psi = psi + dpsi
            diff = abs(psiold-psi)
    
            psiold = psi
    
        Cd       = 0.01385 #From xfoil of the DAE51 at RE=150k, Cl=0.7
        epsilon  = Cd/Cl
        deltar   = (r[1]-r[0])
        thrust   = rho[:,0]*B*np.transpose(np.sum(Gamma*(Wt-epsilon*Wa)*deltar,axis=1))   #T
        torque   = rho[:,0]*B*np.sum(Gamma*(Wa+epsilon*Wt)*r*deltar,axis=1) #Q
        power    = torque*omega       
       
        D        = 2*R
        Cp       = power/(rho[:,0]*(n**3)*(D**5))
        
        etap     = V*thrust/(power)
        print('Propeller Efficiency')
        print(etap)

        return thrust, torque, power, Cp
    
# ----------------------------------------------------------------------
#  Electronic Speed Controller Class
# ----------------------------------------------------------------------

class ESC(Energy_Component):
    
    def __defaults__(self):
        
        self.eff = 0.0
    
    def voltageout(self,conditions):
        """ The electronic speed controllers voltage out
            
            Inputs:
                eta - [0-1] throttle setting
                self.inputs.voltage() - a function that returns volts into the ESC
               
            Outputs:
                voltage out of the ESC
               
            Assumptions:
                The ESC's output voltage is linearly related to throttle setting
               
        """
        eta = conditions.propulsion.throttle
        for ii in range(len(eta)):
            if eta[ii]<=0.0:
                eta[ii] = 0.0
        voltsin = self.inputs.voltagein
        voltsout= eta*voltsin
        
        self.outputs.voltageout = voltsout
        
        return voltsout
    
    def currentin(self):
        """ The current going in
            
            Inputs:
                eff - [0-1] efficiency of the ESC
                self.inputs.power() - a function that returns power
               
            Outputs:
                Current into the ESC
               
            Assumptions:
                The ESC draws current.
               
        """
        eff = self.eff
        currentout = self.inputs.currentout
        currentin  = currentout/eff
        
        self.outputs.currentin = currentin
        
        return currentin
    
# ----------------------------------------------------------------------
#  Avionics Class
# ----------------------------------------------------------------------    

class Avionics(Energy_Component):
    
    def __defaults__(self):
        
        self.draw = 0.0
        
    def power(self):
        """ The avionics input power
            
            Inputs:
                draw
               
            Outputs:
                power output
               
            Assumptions:
                This device just draws power
               
        """
        self.outputs.power = self.draw
        
        return self.draw
    
# ----------------------------------------------------------------------
#  Payload Class
# ----------------------------------------------------------------------  
    
class Payload(Energy_Component):
    
    def __defaults__(self):
        
        self.draw = 0.0
        
    def power(self):
        """ The avionics input power
            
            Inputs:
                draw
               
            Outputs:
                power output
               
            Assumptions:
                This device just draws power
               
        """
        self.outputs.power = self.draw
        
        return self.draw 

# ----------------------------------------------------------------------
#  Solar Logic Class
# ----------------------------------------------------------------------
    
class Solar_Logic(Energy_Component):
    
    def __defaults__(self):
        
        self.MPPTeff       = 0.0
        self.systemvoltage = 0.0
    
    def voltage(self):
        """ The system voltage
            
            Inputs:
                voltage
               
            Outputs:
                voltage
               
            Assumptions:
                this function practically does nothing
               
        """
        volts = self.systemvoltage
        
        self.outputs.systemvoltage = volts
        
        return volts

    def logic(self,conditions,numerics):
        """ The power being sent to the battery
            
            Inputs:
                payload power
                avionics power
                current to the esc
                voltage of the system
                MPPT efficiency
               
            Outputs:
                power to the battery
               
            Assumptions:
                Many: the system voltage is constant, the maximum power
                point is at a constant voltage
               
        """
        #Unpack
        pin        = self.inputs.powerin[:,0]
        pavionics  = self.inputs.pavionics
        ppayload   = self.inputs.ppayload
        volts      = self.voltage()
        esccurrent = self.inputs.currentesc
        I          = numerics.integrate_time
        
        pavail = pin*self.MPPTeff
        
        plevel = pavail -pavionics -ppayload - volts*esccurrent
        
        #Integrate the plevel over time to assess the energy consumption
        #or energy storage
        e = np.dot(I,plevel)
        
        #Send or take power out of the battery
        batlogic      = Data()
        batlogic.pbat = plevel
        batlogic.Ibat = abs(plevel/volts)
        batlogic.e    = e
        
        self.outputs.batlogic = batlogic
        
        return 

# ----------------------------------------------------------------------
#  Battery Class
# ----------------------------------------------------------------------    

class Battery(Energy_Component):
    
    def __defaults__(self):
        
        self.type = 'Li-Ion'
        self.mass = 0.0
        self.CurrentEnergy = 0.0
        self.R0 = 0.07446
        
    def max_energy(self):
        """ The maximum energy the battery can hold
            
            Inputs:
                battery mass
                battery type
               
            Outputs:
                maximum energy the battery can hold
               
            Assumptions:
                This is a simple battery
               
        """
        
        #These need to be fixed
        if self.type=='Li-Ion':
            return 0.90*(10**6)*self.mass
        
        elif self.type=='Li-Po':
            return 0.90*(10**6)*self.mass
        
        elif self.type=='Li-S':
            return 500.*3600.*self.mass        
    
    def energy_calc(self,numerics):
        
        #Unpack
        Ibat  = self.inputs.batlogic.Ibat
        pbat  = self.inputs.batlogic.pbat
        edraw = self.inputs.batlogic.e
        
        Rbat  = self.R0
        I     = numerics.integrate_time
        
        #X value from Mike V.'s battery model
        x = np.divide(self.CurrentEnergy,self.max_energy())[:,0]
        
        #C rate from Mike V.'s battery model
        C = 3600.*pbat/self.max_energy()
        
        f = 1-np.exp(-20*x)-np.exp(-20*(1-x)) #empirical value for discharge
        
        for ii in range(0,len(Ibat)):
            if x[ii]<0:           #reduce discharge losses when model no longer makes sense
                f[ii]=0
        R = self.R0*(1+C*f)       #model discharge characteristics based on changing resistance
        Ploss = (Ibat**2)*R       #calculate resistive losses

        eloss = np.dot(I,Ploss)
        
        #Skip the first energy, since it should already be known
        for ii in range(1,len(Ibat)):
            if pbat[ii]!=0:
                self.CurrentEnergy[ii]=self.CurrentEnergy[ii-1]-edraw[ii]-eloss[ii]
    
            if pbat[ii]<0:
                self.CurrentEnergy[ii]=self.CurrentEnergy[ii-1]-eloss[ii]
                if self.CurrentEnergy[ii]>self.max_energy():
                    self.CurrentEnergy[ii]=self.max_energy()
                
            if self.CurrentEnergy[ii]<0:
                pass
                #print 'Warning, battery out of energy'  
                #Do nothing really!
                
        return  

# the network
class Network(Data):
    def __defaults__(self):
        self.solar_flux  = None
        self.solar_panel = None
        self.motor       = None
        self.propeller   = None
        self.esc         = None
        self.avionics    = None
        self.payload     = None
        self.solar_logic = None
        self.battery     = None
        self.nacelle_dia = 0.0
        self.tag         = 'Network'
    
    # manage process with a driver function
    def evaluate(self,eta,conditions,numerics):
    
        # unpack
        solar_flux  = self.solar_flux
        solar_panel = self.solar_panel
        motor       = self.motor
        propeller   = self.propeller
        esc         = self.esc
        avionics    = self.avionics
        payload     = self.payload
        solar_logic = self.solar_logic
        battery     = self.battery
        
        conditions.frames.planet          = Data()
        conditions.frames.planet.lat      = 37.4300
        conditions.frames.planet.lon      = -122.1700
        conditions.frames.planet.timedate = time.strptime("Sat, Jun 21 8:30:00  2014", "%a, %b %d %H:%M:%S %Y",)  
        #Set battery energy
        battery.CurrentEnergy = battery.max_energy()*np.ones_like(numerics.time)
        
        # step 1
        solar_flux.solar_flux(conditions)
        # link
        solar_panel.inputs.flux = solar_flux.outputs.flux
        # step 2
        solar_panel.power()
        # link
        solar_logic.inputs.powerin = solar_panel.outputs.power
        # step 3
        solar_logic.voltage()
        # link
        esc.inputs.voltagein =  solar_logic.outputs.systemvoltage
        # Step 4
        esc.voltageout(conditions)
        # link
        motor.inputs.voltage = esc.outputs.voltageout 
        # step 5
        motor.omega(conditions)
        # link
        propeller.inputs.omega =  motor.outputs.omega
        #print(motor.outputs.omega)
        # step 6
        F, Q, P, Cplast = propeller.spin(conditions)
        #iterate the Cp here
        
        print('First Iteration')
        diff = abs(Cplast-motor.propCp)
        tol = 1e-5
        
        while (np.any(diff>tol)):
            print('Motor/Prop convergence')
            motor.propCp = Cplast #Change the Cp
            motor.omega(conditions) #Rerun the motor
            propeller.inputs.omega =  motor.outputs.omega #Relink the motor
            F, Q, P, Cplast = propeller.spin(conditions) #Run the motor again
            diff = abs(Cplast-motor.propCp) #Check to see if it converged
            
        #BUT I ONLY USE 1 MOTOR!!!
            
        #Calculate the battery state as a check
        #Run the avionics
        avionics.power()
        # link
        solar_logic.inputs.pavionics =  avionics.outputs.power
        #Run the payload
        payload.power()
        # link
        solar_logic.inputs.ppayload = payload.outputs.power
        #Run the motor for current
        motor.current(conditions)
        # link
        esc.inputs.currentout =  motor.outputs.current
        #Run the esc
        esc.currentin()
        # link
        solar_logic.inputs.currentesc = esc.outputs.currentin
        #
        solar_logic.logic(conditions,numerics)
        # link
        battery.inputs.batlogic = solar_logic.outputs.batlogic
        battery.energy_calc(numerics)
        
        print('Propeller Power Output')
        print P
        
        print('RPM')
        print(motor.outputs.omega*60./(2.*np.pi))
        
        print('Battery Power Output')
        print battery.inputs.batlogic.pbat
        
        print('Solar Input')
        print solar_panel.outputs.power[:,0]
        
        #print(battery.CurrentEnergy)
        
        mdot = np.zeros_like(F)

        #Ok but we have 4 of them!

        F = 4 * F
        P = 4 * P
        
        return F, mdot, P
            
    __call__ = evaluate
