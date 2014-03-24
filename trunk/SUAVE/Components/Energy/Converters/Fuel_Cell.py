""" Fuel_Cell.py: Fuel cell Propulsor Segment; calculates energy consumption and thermodynamic effects of fuel cell """
#by M. Vegh

""" SUAVE.Attrubtes.Components.Energy.Conversion
"""

# ------------------------------------------------------------
#  Imports 
# ------------------------------------------------------------

from SUAVE.Attributes.Propellants import Propellant
from Converter import Converter
from SUAVE.Attributes.Gases import Steam, Air
from SUAVE.Attributes.Propellants import Gaseous_H2
from SUAVE.Attributes.Liquids import Water

# ------------------------------------------------------------
#  Derived Energy Conversion Classes
# ------------------------------------------------------------    

class Fuel_Cell(Converter):
    def __defaults__(self):      
        self.tag = 'Fuel Cell'
        self.propellant = Gaseous_H2()
        self.oxidizer = Air()
        self.MaxPower=0.        #maximum power fuel cell is capable of outputting [W]
        """
        self.tag='Fuel_Cell'
        self.type='H2'                     #only uses a Hydrogen fuel cell for now
        self.Power=0
        self.SpecificPower = 2.08          # kW/kg
        self.eff=0.6                       #stack efficiency
        self.MassDensity =1203.208556      # kg/m^3
        self.Volume = 0.034                # m^3
        self.Fuel = Propellant()
        """
        
        print self.propellant.tag
        if self.propellant.tag=='H2 Gas':
            self.efficiency=.65     #normal fuel cell operating efficiency at sea level
            self.SpecificPower=2.08   #specific power of fuel cell [kW/kg]; default is Nissan 2011 level
            self.MassDensity=1203.208556  #take default as specs from Nissan 2011 fuel cell            
            self.Volume=.034

            
			
        else:
            self.SpecificPower=0.0 
            self.eff=0.0
            self.MassDensity=0.0
            self.Volume=0.0
            
            
    def __call__(self,power,thermo=0,mdot_in=0):

        """  Fuel_Cell(): populates final p_t and T_t values based on initial p_t, T_t values and efficiency
    
         Inputs:    
                    power=Power required for time step (W) (required)
         Outputs:  
                    mdot=mass flow rate of fuel from fuel cell (kg/s)


        """

        # unpack
        try:
            self.i
            i=self.i
            f=self.f
        except:
            AttributeError
  
        
        mdot_water=0                                                                           #predefine variables for return
        mdot_air=0
        if self.active:
            if power.any()>self.MaxPower:
                print "Warning, maximum power output of fuel cell exceeded"
                
          
            if self.propellant.tag=='H2 Gas' or self.propellant.tag=='H2' :
                o_f=8/self.oxidizer.Composition['O2'] 
                mdot=power/(self.propellant.specific_energy*self.efficiency)                      #mass flow rate of the fuel  
                if thermo!=0:
                    gamma =thermo.gamma[i]
                    cp=thermo.cp[i]
                    water=Water()
                    steam=Steam()   
                    cp_water=water.compute_cp(thermo.Tt[i], thermo.pt[i])
                    gamma_steam=steam.compute_gamma(thermo.Tt[i], thermo.pt[i])
                    o_f=8/self.oxidizer.Composition['O2']                                         #o/f mass ratio based on stoichiometry
                    h_vap=water.h_vap                                                             #heat of vaporization of water (J/kg)               
                    mdot_water=mdot*(1+8)                                                         #mass flow rate of water products
                    if mdot_in==0:                                                                #no mass flow rate assigned; assume stoichiometric 
                        mdot_air=mdot*o_f-8*mdot        
                    else:
                        mdot_air=mdot_in-8*mdot                                                   #mass flow rate of leftover air
                    if mdot_in<mdot*o_f and mdot_in !=0:
                        print 'Warning; oxidizer flow rate too small to match power needs'
                
                    R=(mdot_water*steam.R+mdot_air*self.oxidizer.R)/(mdot_water+mdot_air)
                    heat=mdot*self.propellant.specific_energy*(1-self.efficiency)                 #energy lost as heat
                    heat_raise=mdot_water*cp_water*(water.T_vap-thermo.Tt[i])                     #heat energy needed to raise temperature to boiling
 
                
                
              
                    if heat_raise>0 and heat>heat_raise:                                          #all of the products are able to reach the boiling point
                        heat_vap=heat-heat_raise                                                  #heat used for vaporization
                        heat_raise_gas=(heat-heat_raise)-mdot_water*h_vap                         #leftover heat used to raise the temperature of the steam products   
                    
                        if heat_raise_gas>0:
                        
                            cp_steam=steam.compute_cp(water.T_vap, thermo.pt[i])
                        
                            Tf_steam=water.T_vap+heat_raise_gas/(cp_steam*mdot_water)
                            if Tf_steam>500:                                                      #temperature raised by large enough to significantly alter cp 
                                cp_steam2=steam.compute_cp(Tf_steam,thermo.pt[i])
                                cp_steam=(cp_steam+cp_steam2)/2
                                Tf_steam=water.T_vap+heat_raise_gas/(cp_steam*mdot_water)
                            
                            thermo.ht[f]=(Tf_steam*cp_steam*mdot_water+mdot_air*cp*thermo.Tt[i])/(mdot_water+mdot_air)
                            thermo.cp[f]=(mdot_water*cp_steam+mdot_air*cp)/(mdot_water+mdot_air)
                        
                            thermo.Tt[f]=thermo.ht[f]/thermo.cp[f]
                            thermo.gamma[f]=thermo.cp[f]/(thermo.cp[f]-R)
                            thermo.pt[f]=thermo.pt[i]
                       
                        elif heat_raise_gas<0:                                                   #not everything is vaporized
                               
                            cp_steam=steam.compute_cp(water.T_vap, thermo.pt[i])
                            mdot_steam=heat_vap/h_vap
                            mdot_water=mdot_water-mdot_steam
                            thermo.ht[f]=(mdot_steam*cp_steam*373.15+mdot_air*cp*thermo.Tt[i]+mdot_water*cp_water*373.15)/(mdot_water+mdot_air+mdot_steam)
                            thermo.cp[f]=(mdot_water*cp_water+mdot_steam*cp_steam+mdot_air*cp)/(mdot_water+mdot_steam+mdot_air)
                            thermo.Tt[f]=thermo.ht[f]/thermo.cp[f]
                            thermo.pt[f] = thermo.pt[i]
                            thermo.gamma[f]=thermo.cp[f]/(thermo.cp[f]-R)
                        
                    elif heat<heat_raise :                                                      #not able to raise the products to boiling
                        Tt_water=Tt[i]+heat/cp_water
                        thermo.cp[f]=(mdot_water*cp_water+mdot_air*cp)/(mdot_water+mdot_air)    
                        thermo.ht[f]=(mdot_water*Tt_water*cp_water+mdot_air*cp*Tt[i])/(mdot_water+mdot_air)
                        thermo.Tt[f]=thermo.ht[f]/thermo.cp[f]
                        thermo.pt[f] = thermo.pt[i]
                        thermo.gamma[f]=thermo.cp[f]/(thermo.cp[f]-R)
                    elif thermo.Tt[i]>=water.T_vap:                                            #incoming gas already at boiling point

                        heat_raise_gas=heat-mdot_water*h_vap    
                        cp_steam=steam.compute_cp(water.T_vap, thermo.pt[i])                
                        Tf_steam=water.T_vap+heat_raise_gas/(cp_steam*mdot_water)
                        if Tf_steam>500:                                                      #temperature raised by large enough to significantly alter cp 
                            cp_steam2=steam.compute_cp(Tf_steam,thermo.pt[i])
                            cp_steam=(cp_steam+cp_steam2)/2
                            Tf_steam=water.T_vap+heat_raise_gas/(cp_steam*mdot_water)                           
                            thermo.ht[f]=(Tf_steam*cp_steam*mdot_water+mdot_air*cp*thermo.Tt[i])/(mdot_water+mdot_air)
                            thermo.cp[f]=(mdot_water*cp_steam+mdot_air*cp)/(mdot_water+mdot_air)
                            thermo.Tt[f]=thermo.ht[f]/thermo.cp[f]
                            thermo.gamma[f]=thermo.cp[f]/(thermo.cp[f]-R)
                            thermo.pt[f]=thermo.pt[i]
                else:
                     mdot_air=mdot*o_f-8*mdot
        # inactive, add dummy segment
        else:
            thermo.ht[f] = thermo.ht[i]
            thermo.Tt[f] = thermo.Tt[i]
            thermo.pt[f] = thermo.pt[i]
        
        return[mdot, mdot_water+mdot_air]
			