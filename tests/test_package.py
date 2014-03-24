
import SUAVE

a = SUAVE.Structure.Data()

print a

w = SUAVE.Components.Wings.Wing()
for c in w.get_bases():
    print c.__name__


wait = 0
