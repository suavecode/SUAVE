""" convert_distance.py: distance unit converter """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def convert_distance(value,input_unit,output_unit):
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
