


import SUAVE
import json, yaml

data = SUAVE.Core.Data()

data.x = 'x'
data.y = 'y'
data.sub = SUAVE.Core.Data()
data.sub.z = 'z'
data.sub.a = 1

#print json.dumps(data, indent=2)
#print yaml.dump(data.to_dict())



from SUAVE.Core import Data


vehicle = SUAVE.Vehicle()

SUAVE.Input_Output.D3JS.save_tree(vehicle,'tree.json')



