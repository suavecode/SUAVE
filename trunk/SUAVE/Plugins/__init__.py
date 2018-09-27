## @defgroup Plugins
# These are external packages that have been incorporated into SUAVE. SUAVE specific documentation is not used for these packages. 
# Currently the only package used is pint.


from .load_plugin import load_plugin
# these packages are imported by temporarily modifying
# the python path to account for potential absolute
# package imports

pint = load_plugin('pint')