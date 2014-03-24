"""Sizes a Proton Exchange Membrane Fuel Cell based on the maximum power output of the fuel cell"""
#by M.Vegh

""" SUAVE Methods for Energy Systems """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np


# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def size_pem_fc(pemfc,power): 
    """
    Inputs:
    pemfc
    power=power required for the fuel cell [W]
    

    Reads:
    pemfc.rhoc=fuel cell density [kg/m^3]
    pemfc.zeta=porosity coefficient
    pemfc.twall=thickness of cell wall [m]
    pemfc.etac= compressor efficiency
    pemfc.etat= turbine efficiency
    pemfc.A=area of the fuel cell interface [cm^2]
    pemfc.r=area specific resistance [k-Ohm-cm^2]
    pemfc.Eoc=effective activation energy [V]
    pemfc.A1=.slope of the Tafel line (models activation losses) [V]
    pemfc.m=constant in mass-transfer overvoltage equation [V]
    pemfc.n=constant in mass-transfer overvoltage equation

    Outputs:
    pemfc.Ncell=number of fuel cells required
    pemfc.Volume=total volume of the cell[m^3]
    pemfc.Mass_Props.mass=fuel cell mass [kg]
    pemfc.MassDensity=fuel cell volume [kg/m^3]
    pemfc.SpecificPower=fuel cell specific power [kW/kg]
    
    
    """

    #define local versions of variables for convenience

    r=pemfc.r
    A=pemfc.A
    Eoc=pemfc.Eoc
    A1=pemfc.A1
    m=pemfc.m
    n=pemfc.n
    rhoc=pemfc.rhoc
    zeta=pemfc.zeta
    twall=pemfc.twall

    #########################################

    i1=np.linspace(.1,1200.0,200.0) #current density(mA cm^-2): use vector of these with interpolation to find values
    v=Eoc-r*i1-A1*np.log(i1)-m*np.exp(n*i1) #useful voltage vector
    p=np.divide(np.multiply(v,i1),1000.0)*A #obtain power vector in W
    pmax=np.max(p)

  



######
    pemfc.Ncell=round(power/pmax,0)                                            #number of fuel cells required in stack
    pemfc.Volume=pemfc.Ncell*(A*(1.0/np.power(100.0,2.0)))*twall               #total volume of the cell
    pemfc.Mass_Props.mass=pemfc.Volume*rhoc*zeta                               #fuel cell mass in kg
    pemfc.MassDensity=pemfc.Mass_Props.mass/pemfc.Volume                       #fuel cell volume in m^2
    pemfc.SpecificPower=pmax*(1./1000.)/pemfc.Mass_Props.mass                  #fuel cell specific power in kW/kg
    return
    
    
    return 
