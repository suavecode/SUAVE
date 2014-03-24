""" convert_pressure.py: pressure unit converter """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def convert_pressure(value,input_unit,output_unit):
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
