""" convert_length.py: length unit converter """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def convert_length(value,input_unit,output_unit):
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
