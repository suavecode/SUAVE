""" convert_power.py: power unit converter """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def convert_power(value,input_unit,output_unit):
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
