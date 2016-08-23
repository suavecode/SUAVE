""" print_weights.py """

# Created: SUAVE team
# Updated: Carlos Ilario, Feb 2016

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
#  no import

# ----------------------------------------------------------------------
#  Print output file with weight breakdown
# ----------------------------------------------------------------------

def print_weight_breakdown(config,filename = 'weight_breakdown.dat'):
    """ SUAVE.Methods.Results.print_weight_breakdown(config,filename = 'weight_breakdown.dat'):
        
        Print output file with weight breakdown
        
        Inputs:
            config   - data dictionary with the airplane config
            filename [optional] - Name of the file to be created

        Outputs:
            output file

        Assumptions:
			none
		"""
    
    # Imports
    import datetime                 # importing library

    #unpack
    weight_breakdown    = config.weight_breakdown
    mass_properties     = config.mass_properties
    ac_type             = config.systems.accessories
    ctrl_type           = config.systems.control 
    
	# start printing
    fid = open(filename,'w')   # Open output file
    fid.write('Output file with weight breakdown\n\n') #Start output printing
    fid.write( ' DESIGN WEIGHTS \n')    
    if mass_properties.max_takeoff:
        fid.write( ' Maximum Takeoff Weigth ......... : ' + str( '%8.0F' % mass_properties.max_takeoff)    + ' kg\n')
    if mass_properties.max_landing:
        fid.write( ' Maximum Landing Weigth ......... : ' + str( '%8.0F' % mass_properties.max_landing)    + ' kg\n')
    if mass_properties.max_zero_fuel:
        fid.write( ' Maximum Zero Fuel Weigth ....... : ' + str( '%8.0F' % mass_properties.max_zero_fuel)  + ' kg\n')
    if mass_properties.max_fuel:
        fid.write( ' Maximum Fuel Weigth ............ : ' + str( '%8.0F' % mass_properties.max_fuel)       + ' kg\n')
    if mass_properties.max_payload:
        fid.write( ' Maximum Payload Weigth ......... : ' + str( '%8.0F' % mass_properties.max_payload)    + ' kg\n')    
    fid.write('\n')

    fid.write(' ASSUMPTIONS FOR WEIGHT ESTIMATION \n')      
    fid.write( ' Airplane type .................. : ' + ac_type.upper() + '\n')
    fid.write( ' Flight Controls type ........... : ' + ctrl_type.upper() + '\n')    
    fid.write('\n')
    
    fid.write(' EMPTY WEIGHT BREAKDOWN \n')       
    for tag,value in weight_breakdown.items():
        if tag=='payload' or tag=='pax' or tag=='bag' or tag=='fuel' or tag=='empty' or tag=='systems_breakdown':
            continue
        tag = tag.replace('_',' ')
        string = ' ' + tag[0].upper() + tag[1:] + ' '
        string = string.ljust(33,'.') + ' :' 
        fid.write( string + str( '%8.0F'   %   value)  + ' kg\n' )   
    fid.write( ' ........ EMPTY WEIGHT .......... :' + str( '%8.0F' % weight_breakdown.empty)    + ' kg\n') 
    fid.write('\n')
    
    fid.write(' SYSTEMS WEIGHT BREAKDOWN  \n')       
    for tag,value in weight_breakdown.systems_breakdown.items():
        tag = tag.replace('_',' ')
        string = ' ' + tag[0].upper() + tag[1:] + ' '
        string = string.ljust(33,'.') + ' :' 
        fid.write( string + str( '%8.0F'   %   value)  + ' kg\n' )    
    fid.write('\n')    
       

    # Print timestamp
    fid.write('\n'+ 43*'-'+ '\n' + datetime.datetime.now().strftime(" %A, %d. %B %Y %I:%M:%S %p"))
    # done
    fid.close    

# ----------------------------------------------------------------------
#   Module Test
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print(' Error: No test defined ! ')    
