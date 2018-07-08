from math import sin, cos, pi


def create_circle_list(n=80):
    """This creates a list of circle profile points. Profile is defined in y-z plane.

	Assumptions:
	None

	Source:
	N/A

	Inputs:
	n       number of unique points on the circle profile. Another point for closing the profile is added.
	        Thus the full number of points is n+1.

	Outputs:
	x, y, z lists of profile points

	Properties Used:
	N/A
	"""
    x_list = []
    y_list = []
    z_list = []
    for i in range(0, n + 1):
        x = 0.0
        z = sin(360.0 * i / float(n) / 180.0 * pi)
        y = cos(360.0 * i / float(n) / 180.0 * pi)
        z = round(z, 10)
        y = round(y, 10)

        x_list.append(str(x))
        y_list.append(str(y))
        z_list.append(str(z))

    return x_list, y_list, z_list
