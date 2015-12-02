#find_specific_power.py

#Created : M. Vegh, 2014
#Modified: M. Vegh, September 2015
"""determines specific specific power from a ragone curve correlation"""


def find_specific_power(battery, specific_energy):
    const_1=battery.ragone.const_1
    const_2=battery.ragone.const_2
    specific_power=const_1*10.**(const_2*specific_energy)
    battery.specific_power =specific_power
    battery.specific_energy=specific_energy
    
    return