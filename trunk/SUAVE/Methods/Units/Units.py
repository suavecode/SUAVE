# Units.py
#
# Created By:       M. Colonno
# Last updated:     M. Colonno  07/01/13

""" SUAVE Methods for Converting Units
"""

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy
from SUAVE.Structure  import Data
#from SUAVE.Attributes import Constants

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def ConvertPressure(value,input_unit,output_unit):
    """Unit converter for pressure
        
        Inputs:
            value:              value to be converted (float or double)
            input_unit:         input unit (string)
            output_unit:        output unit (string)

            Supported units:    Pa (or Pascals or pascals), 
                                kPa (or Kilopascal or kilopascal), 
                                atm (or Atm or ATM or atmospheres or Atmospheres), 
                                psi (or PSI or lbf/in^2 or lbf/in2 or lbf/sqin), 
                                psf (or PSF or lbf/ft^2 or lbf/ft2 or lbf/sqft)

        Returns:
            converted value
        """

    if input_unit in ["Pa", "Pascals", "pascals"]:
        if output_unit in ["Pa", "Pascals", "pascals"]:
            conversion = 1.0
        elif output_unit in ["psi", "PSI", "lbf/in^2", "lbf/in2", "lbf/sqin"]:
            conversion = 0.000145037738
        elif output_unit in ["psf", "PSF", "lbf/ft^2", "lbf/ft2", "lbf/sqft"]:
            conversion = 0.0208854342
        elif output_unit in ["kPa", "Kilopascal", "kilopascal"]:
            conversion = 1.0e-3
        elif output_unit in ["atm", "Atm", "ATM", "atmospheres", "Atmospheres"]:
            conversion = 9.86923267e-6
        else: 
            print "Unknown output unit " + output_unit + ", no conversion performed."
            conversion = 1.0
    elif input_unit in ["kPa", "Kilopascal", "kilopascal"]:
        if output_unit in ["Pa", "Pascals", "pascals"]:
            conversion = 1000.0
        elif output_unit in ["psi", "PSI", "lbf/in^2", "lbf/in2", "lbf/sqin"]:
            conversion = 0.145037738
        elif output_unit in ["psf", "PSF", "lbf/ft^2", "lbf/ft2", "lbf/sqft"]:
            conversion = 20.8854342
        elif output_unit in ["kPa", "Kilopascal", "kilopascal"]:
            conversion = 1.0
        elif output_unit in ["atm", "Atm", "ATM", "atmospheres", "Atmospheres"]:
            conversion = 0.00986923267
        else: 
            print "Unknown output unit " + output_unit + ", no conversion performed."
            conversion = 1.0
    elif input_unit in ["atm", "Atm", "ATM", "atmospheres", "Atmospheres"]:
        if output_unit in ["Pa", "Pascals", "pascals"]:
            conversion = 101325.0
        elif output_unit in ["psi", "PSI", "lbf/in^2", "lbf/in2", "lbf/sqin"]:
            conversion = 14.6959488
        elif output_unit in ["psf", "PSF", "lbf/ft^2", "lbf/ft2", "lbf/sqft"]:
            conversion = 2116.21662
        elif output_unit in ["kPa", "Kilopascal", "kilopascal"]:
            conversion = 101.325
        elif output_unit in ["atm", "Atm", "ATM", "atmospheres", "Atmospheres"]:
            conversion = 1.0
        else: 
            print "Unknown output unit " + output_unit + ", no conversion performed."
            conversion = 1.0
    elif input_unit in ["psi", "PSI", "lbf/in^2", "lbf/in2", "lbf/sqin"]:
        if output_unit in ["Pa", "Pascals", "pascals"]:
            conversion = 6894.75729
        elif output_unit in ["psi", "PSI", "lbf/in^2", "lbf/in2", "lbf/sqin"]:
            conversion = 1.0
        elif output_unit in ["psf", "PSF", "lbf/ft^2", "lbf/ft2", "lbf/sqft"]:
            conversion = 144.0
        elif output_unit in ["kPa", "Kilopascal", "kilopascal"]:
            conversion = 6.89475729
        elif output_unit in ["atm", "Atm", "ATM", "atmospheres", "Atmospheres"]:
            conversion = 0.0680459639
        else: 
            print "Unknown output unit " + output_unit + ", no conversion performed."
            conversion = 1.0
    elif input_unit in ["psf", "PSF", "lbf/ft^2", "lbf/ft2", "lbf/sqft"]:
        if output_unit in ["Pa", "Pascals", "pascals"]:
            conversion = 47.880259
        elif output_unit in ["psi", "PSI", "lbf/in^2", "lbf/in2", "lbf/sqin"]:
            conversion = 0.00694444444
        elif output_unit in ["psf", "PSF", "lbf/ft^2", "lbf/ft2", "lbf/sqft"]:
            conversion = 1.0
        elif output_unit in ["kPa", "Kilopascal", "kilopascal"]:
            conversion = 0.047880259
        elif output_unit in ["atm", "Atm", "ATM", "atmospheres", "Atmospheres"]:
            conversion = 0.000472541416
        else: 
            print "Unknown output unit " + output_unit + ", no conversion performed."
            conversion = 1.0
    else: 
        print "Unknown input unit " + input_unit + ", no conversion performed."
        conversion = 1.0

    return value*conversion

def ConvertDensity(value,input_unit,output_unit):
    """Unit converter for density
        
        Inputs:
            value:              value to be converted (float or double)
            input_unit:         input unit (string)
            output_unit:        output unit (string)

            Supported units:    kg/m^3 (or kg/m3), 
                                slg/ft^3 (or slg/ft3 or slugs/ft^3 or slugs/ft3), 
                                and lb/ft^3 (or lb/ft3)

        Returns:
            converted value
        """

    if input_unit in ["kg/m^3", "kg/m3"]:
        if output_unit in ["kg/m^3", "kg/m3"]:
            conversion = 1.0
        elif output_unit in ["slg/ft^3", "slg/ft3", "slugs/ft^3", "slugs/ft3"]:
            conversion = 0.00194032033
        elif output_unit in ["lb/ft^3", "lb/ft3"]:
            conversion = 0.0624279606
        else: 
            print "Unknown output unit " + output_unit + ", no conversion performed."
            conversion = 1.0
    elif input_unit in ["slg/ft^3", "slg/ft3", "slugs/ft^3", "slugs/ft3"]:
        if output_unit in ["kg/m^3", "kg/m3"]:
            conversion = 515.378818
        elif output_unit in ["slg/ft^3", "slg/ft3", "slugs/ft^3", "slugs/ft3"]:
            conversion = 1.0 
        elif output_unit in ["lb/ft^3", "lb/ft3"]:
            conversion = 32.1740486
        else: 
            print "Unknown output unit " + output_unit + ", no conversion performed."
            conversion = 1.0
    elif input_unit in ["lb/ft^3", "lb/ft3"]:
        if output_unit in ["kg/m^3", "kg/m3"]:
            conversion = 16.01846337396
        elif output_unit in ["slg/ft^3", "slg/ft3", "slugs/ft^3", "slugs/ft3"]:
            conversion = 0.0310809502
        elif output_unit in ["lb/ft^3", "lb/ft3"]:
            conversion = 1.0
        else: 
            print "Unknown output unit " + output_unit + ", no conversion performed."
            conversion = 1.0
    else: 
        print "Unknown input unit " + input_unit + ", no conversion performed."
        conversion = 1.0

    return value*conversion

def ConvertLength(value,input_unit,output_unit):
    """Unit converter for length
        
        Inputs:
            value:              value to be converted (float or double)
            input_unit:         input unit (string)
            output_unit:        output unit (string)

            Supported units:    mm (or milimeter), 
                                cm (or centimeter), 
                                m (or meter), 
                                ft (or foot or feet), 
                                in (or inch or inches)

        Returns:
            converted value
        """

    if input_unit in ["mm", "milimeter"]:
        if output_unit in ["mm", "milimeter"]:
            conversion = 1.0
        elif output_unit in ["cm", "centimeter"]:
            conversion = 0.1
        elif output_unit in ["m", "meter"]:
            conversion = 0.001
        elif output_unit in ["ft", "foot", "feet"]:
            conversion = 0.003280839895013
        elif output_unit in ["in", "inch", "inches"]:
            conversion = 0.03937007874016
        else: 
            print "Unknown output unit " + output_unit + ", no conversion performed."
            conversion = 1.0
    elif input_unit in ["cm", "centimeter"]:
        if output_unit in ["mm", "milimeter"]:
            conversion = 10.0
        elif output_unit in ["cm", "centimeter"]:
            conversion = 1.0
        elif output_unit in ["m", "meter"]:
            conversion = 0.01
        elif output_unit in ["ft", "foot", "feet"]:
            conversion = 0.03280839895013
        elif output_unit in ["in", "inch", "inches"]:
            conversion = 0.3937007874016
        else: 
            print "Unknown output unit " + output_unit + ", no conversion performed."
            conversion = 1.0
    elif input_unit in ["m", "meter"]:
        if output_unit in ["mm", "milimeter"]:
            conversion = 1000.0 
        elif output_unit in ["cm", "centimeter"]:
            conversion = 100.0
        elif output_unit in ["m", "meter"]:
            conversion = 1.0
        elif output_unit in ["ft", "foot", "feet"]:
            conversion = 3.280839895013
        elif output_unit in ["in", "inch", "inches"]:
            conversion = 39.37007874016
        else: 
            print "Unknown output unit " + output_unit + ", no conversion performed."
            conversion = 1.0
    elif input_unit in ["ft", "foot", "feet"]:
        if output_unit in ["mm", "milimeter"]:
            conversion = 304.8 
        elif output_unit in ["cm", "centimeter"]:
            conversion = 30.48
        elif output_unit in ["m", "meter"]:
            conversion = 0.3048
        elif output_unit in ["ft", "foot", "feet"]:
            conversion = 1.0
        elif output_unit in ["in", "inch", "inches"]:
            conversion = 12.0
        else: 
            print "Unknown output unit " + output_unit + ", no conversion performed."
            conversion = 1.0
    elif input_unit in ["in", "inch", "inches"]:
        if output_unit in ["mm", "milimeter"]:
            conversion = 25.4 
        elif output_unit in ["cm", "centimeter"]:
            conversion = 2.54
        elif output_unit in ["m", "meter"]:
            conversion = 0.0254
        elif output_unit in ["ft", "foot", "feet"]:
            conversion = 0.08333333333333
        elif output_unit in ["in", "inch", "inches"]:
            conversion = 1.0
        else: 
            print "Unknown output unit " + output_unit + ", no conversion performed."
            conversion = 1.0
    else: 
        print "Unknown input unit " + input_unit + ", no conversion performed."
        conversion = 1.0

    return value*conversion

def ConvertTemperature(value,input_unit,output_unit):
    """Unit converter for temperature
        
        Inputs:
            value:              value to be converted (float or double)
            input_unit:         input unit (string)
            output_unit:        output unit (string)

            Supported units:    K (or Kelvin or kelvin), 
                                F (or Fahrenheit or fahrenheit), 
                                C (or Celsius or celsius), 
                                R (or Rankine or rankine)

        Returns:
            converted value
        """

    if input_unit in ["K", "Kelvin", "kelvin"]:
        if output_unit in ["K", "Kelvin", "kelvin"]:
            conversion = 1.0; offset = 0.0
        elif output_unit in ["F", "Fahrenheit", "fahrenheit"]:
            conversion = 1.8; offset = -459.67
        elif output_unit in ["C", "Celsius", "celsius"]:
            conversion = 1.0; offset = -273.15
        elif output_unit in ["R", "Rankine", "rankine"]:
            conversion = 1.8; offset = 0.0 
        else:
            print "Unknown output unit " + output_unit + ", no conversion performed."
            conversion = 1.0
    elif input_unit in ["F", "Fahrenheit", "fahrenheit"]:
        if output_unit in ["K", "Kelvin", "kelvin"]:
            conversion = 5.0/9.0; offset = 459.67*conversion
        elif output_unit in ["F", "Fahrenheit", "fahrenheit"]:
            conversion = 1.0; offset = 0.0
        elif output_unit in ["C", "Celsius", "celsius"]:
            conversion = 5.0/9.0; offset = -32.0*conversion
        elif output_unit in ["R", "Rankine", "rankine"]:
            conversion = 1.0; offset = 459.67
        else: 
            print "Unknown output unit " + output_unit + ", no conversion performed."
            conversion = 1.0
    elif input_unit in ["C", "Celsius", "celsius"]:
        if output_unit in ["K", "Kelvin", "kelvin"]:
            conversion = 1.0; offset = 273.15
        elif output_unit in ["F", "Fahrenheit", "fahrenheit"]:
            conversion = 1.8; offset = 32.0
        elif output_unit in ["C", "Celsius", "celsius"]:
            conversion = 1.0; offset = 0.0
        elif output_unit in ["R", "Rankine", "rankine"]:
            conversion = 1.8; offset = 273.15*conversion
        else: 
            print "Unknown output unit " + output_unit + ", no conversion performed."
            conversion = 1.0
    elif input_unit in ["R", "Rankine", "rankine"]:
        if output_unit in ["K", "Kelvin", "kelvin"]:
            conversion = 5.0/9.0; offset = 0.0
        elif output_unit in ["F", "Fahrenheit", "fahrenheit"]:
            conversion = 1.0; offset = -459.67
        elif output_unit in ["C", "Celsius", "celsius"]:
            conversion = 5.0/9.0; offset = -273.15
        elif output_unit in ["R", "Rankine", "rankine"]:
            conversion = 1.0; offset = 0.0
        else: 
            print "Unknown output unit " + output_unit + ", no conversion performed."
            conversion = 1.0
    else: 
        print "Unknown input unit " + input_unit + ", no conversion performed."
        conversion = 1.0

    return value*conversion + offset

def ConvertDistance(value,input_unit,output_unit):
    """Unit converter for distance
        
        Inputs:
            value:              value to be converted (float or double)
            input_unit:         input unit (string)
            output_unit:        output unit (string)

            Supported units:    km (or KM or kilometer or Kilometer), 
                                m (or mile or Mile), 
                                nm (or nautical mile or Nautical Mile)

        Returns:
            converted value
        """

    if input_unit in ["km", "KM", "kilometer", "Kilometer"]:
        if output_unit in ["km", "KM", "kilometer", "Kilometer"]:
            conversion = 1.0
        elif output_unit in ["m", "mile", "Mile"]:
            conversion = 0.6213711922373
        elif output_unit in ["nm", "nautical mile", "Nautical Mile"]:
            conversion = 0.539957
        else:
            print "Unknown output unit " + output_unit + ", no conversion performed."
            conversion = 1.0
    elif input_unit in ["m", "mile", "Mile"]:
        if output_unit in ["km", "KM", "kilometer", "Kilometer"]:
            conversion = 1.609344
        elif output_unit in ["m", "mile", "Mile"]:
            conversion = 1.0
        elif output_unit in ["nm", "nautical mile", "Nautical Mile"]:
            conversion = 0.868976
        else: 
            print "Unknown output unit " + output_unit + ", no conversion performed."
            conversion = 1.0
    elif input_unit in ["nm", "nautical mile", "Nautical Mile"]:
        if output_unit in ["km", "KM", "kilometer", "Kilometer"]:
            conversion = 1.852
        elif output_unit in ["m", "mile", "Mile"]:
            conversion = 1.15078
        elif output_unit in ["nm", "nautical mile", "Nautical Mile"]:
            conversion = 1.0
        else: 
            print "Unknown output unit " + output_unit + ", no conversion performed."
            conversion = 1.0
    else: 
        print "Unknown input unit " + input_unit + ", no conversion performed."
        conversion = 1.0

    return value*conversion

def ConvertPower(value,input_unit,output_unit):
    """Unit converter for power
        
        Inputs:
            value:              value to be converted (float or double)
            input_unit:         input unit (string)
            output_unit:        output unit (string)

            Supported units:    W (or watts or Watts)
                                kW (or KW or kilowatts or Kilowatts), 
                                hp (or HP or horsepower or Horsepower), 
                                ft-lb/s (or foot-pound/sec)

        Returns:
            converted value
        """

    if input_unit in ["W", "watts", "Watts"]:
        if output_unit in ["W", "watts", "Watts"]:
            conversion = 1.0
        elif output_unit in ["kW", "KW", "kilowatts", "Kilowatts"]:
            conversion = 1.0e-3
        elif output_unit in ["hp", "HP", "horsepower", "Horsepower"]:
            conversion = 1.34102209e-3
        elif output_unit in ["ft-lb/s", "foot-pound/sec"]:
            conversion = 0.737562149
        else:
            print "Unknown output unit " + output_unit + ", no conversion performed."
            conversion = 1.0
    elif input_unit in ["kW", "KW", "kilowatts", "Kilowatts"]:
        if output_unit in ["W", "watts", "Watts"]:
            conversion = 1000.0
        elif output_unit in ["kW", "KW", "kilowatts", "Kilowatts"]:
            conversion = 1.0
        elif output_unit in ["hp", "HP", "horsepower", "Horsepower"]:
            conversion = 1.34102209
        elif output_unit in ["ft-lb/s", "foot-pound/sec"]:
            conversion = 737.562149
        else:
            print "Unknown output unit " + output_unit + ", no conversion performed."
            conversion = 1.0
    elif input_unit in ["hp", "HP", "horsepower", "Horsepower"]:
        if output_unit in ["W", "watts", "Watts"]:
            conversion = 745.699872
        elif output_unit in ["kW", "KW", "kilowatts", "Kilowatts"]:
            conversion = 0.745699872
        elif output_unit in ["hp", "HP", "horsepower", "Horsepower"]:
            conversion = 1.0
        elif output_unit in ["ft-lb/s", "foot-pound/sec"]:
            conversion = 550.0
        else:
            print "Unknown output unit " + output_unit + ", no conversion performed."
            conversion = 1.0
    elif input_unit in ["ft-lb/s", "foot-pound/sec"]:
        if output_unit in ["W", "watts", "Watts"]:
            conversion = 1.35581795
        elif output_unit in ["kW", "KW", "kilowatts", "Kilowatts"]:
            conversion = 1.35581795e-3
        elif output_unit in ["hp", "HP", "horsepower", "Horsepower"]:
            conversion = 1.81818182e-3
        elif output_unit in ["ft-lb/s", "foot-pound/sec"]:
            conversion = 1.0
        else:
            print "Unknown output unit " + output_unit + ", no conversion performed."
            conversion = 1.0
    else: 
        print "Unknown input unit " + input_unit + ", no conversion performed."
        conversion = 1.0

    return value*conversion

def ConvertForce(value,input_unit,output_unit):
    """Unit converter for force
        
        Inputs:
            value:              value to be converted (float or double)
            input_unit:         input unit (string)
            output_unit:        output unit (string)

            Supported units:    N (or Newton or newton), 
                                kN (or KN or Kilonewton or kilonewton), 
                                lbf (or pound or pound-force)

        Returns:
            converted value
        """

    if input_unit in ["N", "Newton", "newton"]:
        if output_unit in ["N", "Newton", "newton"]:
            conversion = 1.0
        elif output_unit in ["kN", "KN", "Kilonewton", "kilonewton"]:
            conversion = 1.0e-3
        elif output_unit in ["lbf", "pound", "pound-force"]:
            conversion = 0.224808943
        else:
            print "Unknown output unit " + output_unit + ", no conversion performed."
            conversion = 1.0
    elif input_unit in ["kN", "KN", "Kilonewton", "kilonewton"]:
        if output_unit in ["N", "Newton", "newton"]:
            conversion = 1000.0
        elif output_unit in ["kN", "KN", "Kilonewton", "kilonewton"]:
            conversion = 1.0
        elif output_unit in ["lbf", "pound", "pound-force"]:
            conversion = 224.808943
        else: 
            print "Unknown output unit " + output_unit + ", no conversion performed."
            conversion = 1.0
    elif input_unit in ["lbf", "pound", "pound-force"]:
        if output_unit in ["N", "Newton", "newton"]:
            conversion = 4.44822162
        elif output_unit in ["kN", "KN", "Kilonewton", "kilonewton"]:
            conversion = 4.44822162e-3
        elif output_unit in ["lbf", "pound", "pound-force"]:
            conversion = 1.0
        else: 
            print "Unknown output unit " + output_unit + ", no conversion performed."
            conversion = 1.0
    else: 
        print "Unknown input unit " + input_unit + ", no conversion performed."
        conversion = 1.0

    return value*conversion