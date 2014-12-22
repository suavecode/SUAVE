# Tim Momose, December 2014

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

#from SUAVE.Structure import Data, Data_Exception, Data_Warning
from purge_files import purge_files

def write_run_cases(avl_inputs):
	
	# unpack avl_inputs
	files_path     = avl_inputs.input_files.reference_path
	cases_path     = files_path + avl_inputs.input_files.cases
	configuration  = avl_inputs.configuration
	kases          = avl_inputs.cases
	
	base_case_text = \
'''

 ---------------------------------------------
 Run case  {0}:   {1}

 alpha        ->  alpha       =   {2}
 beta         ->  beta        =   {3}
 pb/2V        ->  pb/2V       =   0.00000
 qc/2V        ->  qc/2V       =   0.00000
 rb/2V        ->  rb/2V       =   0.00000
{4}

 alpha     =   0.00000     deg
 beta      =   0.00000     deg
 pb/2V     =   0.00000
 qc/2V     =   0.00000
 rb/2V     =   0.00000
 CL        =  0.000000
 CDo       =  {5}
 bank      =   0.00000     deg
 elevation =   0.00000     deg
 heading   =   0.00000     deg
 Mach      =   {6}
 velocity  =   {7}     m/s
 density   =   {8}     kg/m^3
 grav.acc. =   {9}     m/s^2
 turn_rad. =   0.00000     m
 load_fac. =   0.00000
 X_cg      =   {10}     m
 Y_cg      =   {11}     m
 Z_cg      =   {12}     m
 mass      =   {13}     kg
 Ixx       =   {14}     kg-m^2
 Iyy       =   {15}     kg-m^2
 Izz       =   {16}     kg-m^2
 Ixy       =   {17}     kg-m^2
 Iyz       =   {18}     kg-m^2
 Izx       =   {19}     kg-m^2
 visc CL_a =   0.00000
 visc CL_u =   0.00000
 visc CM_a =   0.00000
 visc CM_u =   0.00000

'''#{4} is a set of control surface inputs that will vary depending on the control surface configuration

	
	# Open the geometry file after purging if it already exists
	purge_files([cases_path])
	runcases = open(cases_path,'w')
	
	try:
		CD0         = configuration.parasite_drag
		x_cg        = configuration.reference_values.cg_coords[0]
		y_cg        = configuration.reference_values.cg_coords[1]
		z_cg        = configuration.reference_values.cg_coords[2]
		mass        = configuration.mass_properties.mass
		Ixx         = configuration.mass_properties.inertial.Ixx
		Iyy         = configuration.mass_properties.inertial.Iyy
		Izz         = configuration.mass_properties.inertial.Izz
		Ixy         = configuration.mass_properties.inertial.Ixy
		Iyz         = configuration.mass_properties.inertial.Iyz
		Izx         = configuration.mass_properties.inertial.Izx
		
		for case in kases.cases:
			# Unpack inputs
			index = case.index
			name  = case.tag
			alpha = case.angles.alpha
			beta  = case.angles.beta
			mach  = case.conditions.mach
			v     = case.conditions.v_inf
			rho   = case.conditions.rho
			g     = case.conditions.gravitational_acc
			# form controls text
			controls = []
			for cs in case.control_deflections:
				cs_text = make_controls_case_text(cs)
				controls.append(cs_text)
			controls_text = ''.join(controls)
			case_text = base_case_text.format(index,name,alpha,beta,controls_text,
											  CD0,mach,v,rho,g,x_cg,y_cg,z_cg,mass,
											  Ixx,Iyy,Izz,Ixy,Iyz,Izx)
			runcases.write(case_text)
	
	finally:	# don't leave the file open if something goes wrong
		runcases.close()
	
	return avl_inputs



def make_controls_case_text(control_from_cases):
	
	base_control_cond_text = '{0}      ->  {0}     =   {1}    \n'
	
	# Unpack inputs
	name = control_from_cases.tag
	d    = control_from_cases.deflection

	# Form text
	controls_case_text = base_control_cond_text.format(name,d)

	return controls_case_text

