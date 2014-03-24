""" convert_force.py: force unit converter """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def convert_force(value,input_unit,output_unit):
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