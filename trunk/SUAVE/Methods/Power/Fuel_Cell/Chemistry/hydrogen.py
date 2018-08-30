## @ingroup Methods-Power-Fuel_Cell-Chemistry

# hydrogen.py
#
# Created : ### 2015, M. Vegh 
# Modified: Sep 2015, M. Vegh
#           Feb 2016, E. Botero

# ----------------------------------------------------------------------
#  Hydrogen
# ----------------------------------------------------------------------

## @ingroup Methods-Power-Fuel_Cell-Chemistry
def hydrogen(fuel_cell,conditions,numerics):
    """ Calculates chemical properties of fuel cell exhaust
     
       Assumptions: 
       Stoichiometric reaction
       
   
       Inputs:
       fuel_cell.
         efficiency                     [dimensionless]
         inputs.
           power_in                     [Watts]
         propellant.
           tag
           specific_energy              [J/kg]
         oxidizer.
           Composition.
             O2
       thermo.
         gamma                          [dimensionless]
         cp                             [J/kg-K]
         Tt(to be modified)             [K]
         Pt(to be modified)             [Pa]
       
       Outputs:
         mdot                           [kg/s]
         mdot+mdot_air                  [kg/s]
        


    """
    power  = fuel_cell.inputs.power_in
    thermo = conditions.thermo
    if fuel_cell.active:
        if power.any()>fuel_cell.max_power:
            print("Warning, maximum power output of fuel cell exceeded")
                
            if fuel_cell.propellant.tag=='H2 Gas' or fuel_cell.propellant.tag=='H2':
                
                o_f = 8/fuel_cell.oxidizer.Composition['O2'] 
                
                # mass flow rate of the fuel  
                mdot = power/(fuel_cell.propellant.specific_energy*fuel_cell.efficiency)  
                
                if thermo!=0:
                    gamma = thermo.gamma[i]
                    cp    = thermo.cp[i]
                    water = Water()
                    steam = Steam()
                    
                    cp_water    = water.compute_cp(thermo.Tt[i], thermo.pt[i])
                    gamma_steam = steam.compute_gamma(thermo.Tt[i], thermo.pt[i])
                    
                    # o/f mass ratio based on stoichiometry
                    o_f = 8/fuel_cell.oxidizer.Composition['O2']       
                    
                    # heat of vaporization of water (J/kg) 
                    h_vap = water.h_vap    
                    
                    # mass flow rate of water products
                    mdot_water=mdot*(1+8)   
                    
                    if mdot_in==0: 
                        #no mass flow rate assigned; assume stoichiometric 
                        mdot_air = mdot*o_f-8*mdot        
                    else:
                        #mass flow rate of leftover air
                        mdot_air = mdot_in-8*mdot       
                        
                    if mdot_in<mdot*o_f and mdot_in !=0:
                        print('Warning; oxidizer flow rate too small to match power needs')
                
                    R = (mdot_water*steam.R+mdot_air*fuel_cell.oxidizer.R)/(mdot_water+mdot_air)
                    
                    #energy lost as heat
                    heat       = mdot*fuel_cell.propellant.specific_energy*(1-fuel_cell.efficiency)  
                    
                    #heat energy needed to raise temperature to boiling
                    heat_raise = mdot_water*cp_water*(water.T_vap-thermo.Tt[i])                     
              
                    #all of the products are able to reach the boiling point
                    if heat_raise>0 and heat>heat_raise:     
                        
                        # heat used for vaporization
                        heat_vap       = heat-heat_raise                  
                        
                        #leftover heat used to raise the temperature of the steam products   
                        heat_raise_gas = (heat-heat_raise)-mdot_water*h_vap                         
                    
                        if heat_raise_gas>0:
                        
                            cp_steam = steam.compute_cp(water.T_vap, thermo.pt[i])
                            Tf_steam = water.T_vap+heat_raise_gas/(cp_steam*mdot_water)
                            
                            #temperature raised by large enough to significantly alter cp 
                            if Tf_steam>500:                                                      
                                cp_steam2 = steam.compute_cp(Tf_steam,thermo.pt[i])
                                cp_steam  =(cp_steam+cp_steam2)/2
                                Tf_steam  = water.T_vap+heat_raise_gas/(cp_steam*mdot_water)
                            
                            thermo.ht[f] = (Tf_steam*cp_steam*mdot_water+mdot_air*cp*thermo.Tt[i])/(mdot_water+mdot_air)
                            thermo.cp[f] = (mdot_water*cp_steam+mdot_air*cp)/(mdot_water+mdot_air)
                        
                            thermo.Tt[f]    = thermo.ht[f]/thermo.cp[f]
                            thermo.gamma[f] = thermo.cp[f]/(thermo.cp[f]-R)
                            thermo.pt[f]    = thermo.pt[i]
                       
                        #not everything is vaporized
                        elif heat_raise_gas<0:                                                   
                               
                            cp_steam   = steam.compute_cp(water.T_vap, thermo.pt[i])
                            mdot_steam = heat_vap/h_vap
                            mdot_water = mdot_water-mdot_steam
                            
                            thermo.ht[f]    = (mdot_steam*cp_steam*373.15+mdot_air*cp*thermo.Tt[i]+\
                                               mdot_water*cp_water*373.15)/(mdot_water+mdot_air+mdot_steam)
                            thermo.cp[f]    = (mdot_water*cp_water+mdot_steam*cp_steam+mdot_air*cp)/(mdot_water \
                                                                                                     +mdot_steam+mdot_air)
                            thermo.Tt[f]    = thermo.ht[f]/thermo.cp[f]
                            thermo.pt[f]    = thermo.pt[i]
                            thermo.gamma[f] = thermo.cp[f]/(thermo.cp[f]-R)
                        
                    #not able to raise the products to boiling
                    elif heat<heat_raise:             
                        
                        Tt_water = Tt[i]+heat/cp_water
                        
                        thermo.cp[f]    = (mdot_water*cp_water+mdot_air*cp)/(mdot_water+mdot_air)    
                        thermo.ht[f]    = (mdot_water*Tt_water*cp_water+mdot_air*cp*Tt[i])/(mdot_water+mdot_air)
                        thermo.Tt[f]    = thermo.ht[f]/thermo.cp[f]
                        thermo.pt[f]    = thermo.pt[i]
                        thermo.gamma[f] = thermo.cp[f]/(thermo.cp[f]-R)
                        
                    #incoming gas already at boiling point
                    elif thermo.Tt[i]>=water.T_vap:                                            

                        heat_raise_gas = heat-mdot_water*h_vap    
                        cp_steam       = steam.compute_cp(water.T_vap, thermo.pt[i])                
                        Tf_steam       = water.T_vap+heat_raise_gas/(cp_steam*mdot_water)
                        
                        #temperature raised by large enough to significantly alter cp 
                        if Tf_steam>500:                                                      
                            cp_steam2 = steam.compute_cp(Tf_steam,thermo.pt[i])
                            cp_steam  = (cp_steam+cp_steam2)/2
                            Tf_steam  = water.T_vap+heat_raise_gas/(cp_steam*mdot_water)  
                            
                            thermo.ht[f]=(Tf_steam*cp_steam*mdot_water+mdot_air*cp*thermo.Tt[i])/(mdot_water+mdot_air)
                            thermo.cp[f]=(mdot_water*cp_steam+mdot_air*cp)/(mdot_water+mdot_air)
                            thermo.Tt[f]=thermo.ht[f]/thermo.cp[f]
                            thermo.gamma[f]=thermo.cp[f]/(thermo.cp[f]-R)
                            thermo.pt[f]=thermo.pt[i]
                else:
                    mdot_air = mdot*o_f-8*mdot
                    
        else:
            thermo.ht[f] = thermo.ht[i]
            thermo.Tt[f] = thermo.Tt[i]
            thermo.pt[f] = thermo.pt[i]

    return [mdot, mdot_water+mdot_air]