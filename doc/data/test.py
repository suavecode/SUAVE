


import MARC
import json, yaml

data = MARC.Core.Data()

data.x = 'x'
data.y = 'y'
data.sub = MARC.Core.Data()
data.sub.z = 'z'
data.sub.a = 1

#print json.dumps(data, indent=2)
#print yaml.dump(data.to_dict())



from MARC.Core import Data


vehicle = MARC.Vehicle()

MARC.Input_Output.D3JS.save_tree(vehicle,'tree.json')



