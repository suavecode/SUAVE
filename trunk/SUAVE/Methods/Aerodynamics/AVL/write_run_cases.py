# write_runcases.py
# 
# Created:  Dec 2014, T. Momose
# Modified: Jan 2016, E. Botero
# Modified: Apr 2017, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from purge_files import purge_files

def write_run_cases(avl_object):


    # unpack avl_inputs
    batch_filename = avl_object.current_status.batch_file
    aircraft       = avl_object.geometry

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
 CL        =   0.00000
 CDo       =   0.00000
 bank      =   0.00000     deg
 elevation =   0.00000     deg
 heading   =   0.00000     deg
 Mach      =   {5}
 velocity  =   {6}     m/s
 density   =   {7}     kg/m^3
 grav.acc. =   {8}     m/s^2
 turn_rad. =   0.00000     m
 load_fac. =   0.00000
 X_cg      =   {9}     m
 Y_cg      =   {10}     m
 Z_cg      =   {11}     m
 mass      =   {12}     kg
 Ixx       =   {13}     kg-m^2
 Iyy       =   {14}     kg-m^2
 Izz       =   {15}     kg-m^2
 Ixy       =   {16}     kg-m^2
 Iyz       =   {17}     kg-m^2
 Izx       =   {18}     kg-m^2
 visc CL_a =   0.00000
 visc CL_u =   0.00000
 visc CM_a =   0.00000
 visc CM_u =   0.00000

'''#{4} is a set of control surface inputs that will vary depending on the control surface configuration

    # Open the geometry file after purging if it already exists
    purge_files([batch_filename])
    with open(batch_filename,'w') as runcases:

        x_cg = aircraft.mass_properties.center_of_gravity[0]
        y_cg = aircraft.mass_properties.center_of_gravity[1]
        z_cg = aircraft.mass_properties.center_of_gravity[2]
        mass = aircraft.mass_properties.max_takeoff 
        moments_of_inertia = aircraft.mass_properties.moments_of_inertia.tensor
        Ixx  = moments_of_inertia[0][0]
        Iyy  = moments_of_inertia[1][1]
        Izz  = moments_of_inertia[2][2]
        Ixy  = moments_of_inertia[0][1]
        Iyz  = moments_of_inertia[1][2]
        Izx  = moments_of_inertia[2][0]

        for case in avl_object.current_status.cases:
            index = case.index
            name  = case.tag
            alpha = case.conditions.aerodynamics.angle_of_attack
            beta  = case.conditions.aerodynamics.side_slip_angle
            mach  = case.conditions.freestream.mach
            v     = case.conditions.freestream.velocity
            rho   = case.conditions.freestream.density
            g     = case.conditions.freestream.gravitational_acceleration

            # form controls text
            controls = []
            if case.stability_and_control.control_deflections:
                for cs in case.stability_and_control.control_deflections:
                    cs_text = make_controls_case_text(cs)
                    controls.append(cs_text)
            controls_text = ''.join(controls)
            case_text = base_case_text.format(index,name,alpha,beta,controls_text,
                                              mach,v,rho,g,x_cg,y_cg,z_cg,mass,
                                              Ixx,Iyy,Izz,Ixy,Iyz,Izx)
            runcases.write(case_text)

    return


def make_controls_case_text(control_deflection):

    base_control_cond_text = '{0}      ->  {0}     =   {1}    \n'

    # Unpack inputs
    name = control_deflection.tag
    d    = control_deflection.deflection

    # Form text
    controls_case_text = base_control_cond_text.format(name,d)

    return controls_case_text
