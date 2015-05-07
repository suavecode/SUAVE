# Aliasing.py
# 
# Created:  May 2015, E. Botero
# Modified: 

# ----------------------------------------------------------------------        
#   Aliases
# ----------------------------------------------------------------------    

def setup_aliases():
    
    # 'alias' , ['data.path1.name','data.path2.name']
    aliases = [
        'aspect_ratio'  ,  [
            'configs.*.wings.main_wing.aspect_ratio'],
        'reference_area',  [
            'configs.*.wings.main_wing.reference_area'],
        'sweep'         ,  [
            'configs.*.wings.main_wing.sweep'],
        'design_thrust' ,  [
            'configs.*.propulsors.turbo_fan.design_thrust'],
        'wing_thickness',  [
            'configs.*.wings.main_wing.thickness'],
        'MTOW'          ,  [
            'configs.*.mass_properties.max_takeoff',
            'configs.*.mass_properties.takeoff'],
        'MZFW'          ,  [
       'configs.*.mass_properties.max_zero_fuel']
        ]
    
    return aliases
        