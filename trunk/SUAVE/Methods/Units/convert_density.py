""" convert_density.py: density unit converter """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def convert_density(value,input_unit,output_unit):
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
