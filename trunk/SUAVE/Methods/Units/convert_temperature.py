""" convert_temperature.py: temperature unit converter """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def convert_temperature(value,input_unit,output_unit):
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
