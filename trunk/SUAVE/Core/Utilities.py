## @ingroup Core
# Utilities.py
#
# Created:  Oct 2022, M. Clarke

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import numpy as np
import jax.numpy as jnp
from jax import  jit 
#from tensorflow_probability.substrates import jax as tfp
import scipy.special as sp
#import tensorflow as tf
#from jax._src import api
#from jax.experimental import host_callback as hcb

#from tensorflow.python.ops.special_math_ops import fresnel_sin, fresnel_cos

import jax
 
def interp2d(x,y,xp,yp,zp,fill_value= None):
    """
    Bilinear interpolation on a grid. ``CartesianGrid`` is much faster if the data
    lies on a regular grid.
    Args:
        x, y: 1D arrays of point at which to interpolate. Any out-of-bounds
            coordinates will be clamped to lie in-bounds.
        xp, yp: 1D arrays of points specifying grid points where function values
            are provided.
        zp: 2D array of function values. For a function `f(x, y)` this must
            satisfy `zp[i, j] = f(xp[i], yp[j])`
    Returns:
        1D array `z` satisfying `z[i] = f(x[i], y[i])`.
    """
    #if xp.ndim != 1 or yp.ndim != 1:
        #raise ValueError("xp and yp must be 1D arrays")
    #if zp.shape != (xp.shape + yp.shape):
        #raise ValueError("zp must be a 2D array with shape xp.shape + yp.shape")

    ix = np.clip(np.searchsorted(xp, x, side="right"), 1, len(xp) - 1)
    iy = np.clip(np.searchsorted(yp, y, side="right"), 1, len(yp) - 1)

    # Using Wikipedia's notation (https://en.wikipedia.org/wiki/Bilinear_interpolation)
    z_11 = zp[ix - 1, iy - 1]
    z_21 = zp[ix, iy - 1]
    z_12 = zp[ix - 1, iy]
    z_22 = zp[ix, iy]

    z_xy1 = (xp[ix] - x) / (xp[ix] - xp[ix - 1]) * z_11 + (x - xp[ix - 1]) / (
        xp[ix] - xp[ix - 1]
    ) * z_21
    z_xy2 = (xp[ix] - x) / (xp[ix] - xp[ix - 1]) * z_12 + (x - xp[ix - 1]) / (
        xp[ix] - xp[ix - 1]
    ) * z_22

    z = (yp[iy] - y) / (yp[iy] - yp[iy - 1]) * z_xy1 + (y - yp[iy - 1]) / (
        yp[iy] - yp[iy - 1]
    ) * z_xy2

    if fill_value is not None:
        oob = np.logical_or(
            x < xp[0], np.logical_or(x > xp[-1], np.logical_or(y < yp[0], y > yp[-1]))
        )
        z = np.where(oob, fill_value, z)

    return z

@jit
def jax_interp2d(x,y,xp,yp,zp,fill_value= None):
    """
    Bilinear interpolation on a grid. ``CartesianGrid`` is much faster if the data
    lies on a regular grid.
    Args:
        x, y: 1D arrays of point at which to interpolate. Any out-of-bounds
            coordinates will be clamped to lie in-bounds.
        xp, yp: 1D arrays of points specifying grid points where function values
            are provided.
        zp: 2D array of function values. For a function `f(x, y)` this must
            satisfy `zp[i, j] = f(xp[i], yp[j])`
    Returns:
        1D array `z` satisfying `z[i] = f(x[i], y[i])`.
    """
    #if xp.ndim != 1 or yp.ndim != 1:
        #raise ValueError("xp and yp must be 1D arrays")
    #if zp.shape != (xp.shape + yp.shape):
        #raise ValueError("zp must be a 2D array with shape xp.shape + yp.shape")

    ix = jnp.clip(jnp.searchsorted(xp, x, side="right"), 1, len(xp) - 1)
    iy = jnp.clip(jnp.searchsorted(yp, y, side="right"), 1, len(yp) - 1)

    # Using Wikipedia's notation (https://en.wikipedia.org/wiki/Bilinear_interpolation)
    z_11 = zp[ix - 1, iy - 1]
    z_21 = zp[ix, iy - 1]
    z_12 = zp[ix - 1, iy]
    z_22 = zp[ix, iy]

    z_xy1 = (xp[ix] - x) / (xp[ix] - xp[ix - 1]) * z_11 + (x - xp[ix - 1]) / (
        xp[ix] - xp[ix - 1]
    ) * z_21
    z_xy2 = (xp[ix] - x) / (xp[ix] - xp[ix - 1]) * z_12 + (x - xp[ix - 1]) / (
        xp[ix] - xp[ix - 1]
    ) * z_22

    z = (yp[iy] - y) / (yp[iy] - yp[iy - 1]) * z_xy1 + (y - yp[iy - 1]) / (
        yp[iy] - yp[iy - 1]
    ) * z_xy2

    if fill_value is not None:
        oob = jnp.logical_or(
            x < xp[0], jnp.logical_or(x > xp[-1], jnp.logical_or(y < yp[0], y > yp[-1]))
        )
        z = jnp.where(oob, fill_value, z)

    return z

#@jit
#def jjv(v,z):
    #v = jnp.array(v,dtype=jnp.float32)
    #z = jnp.array(z,dtype=jnp.float32)
    
    #I_ve_exp = tfp.math.bessel_ive(v,z) # exponentially scaled version of the modified Bessel function of the first kind # https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/math/bessel_ive  
    #I_ve     = I_ve_exp/jnp.exp(-abs(z))# unscaling 
    ##jjv_val = jnp.exp((v*jnp.pi*1j)/2)*I_ve
    #jjv_val  = (1j**(-v))*I_ve  # https://proofwiki.org/wiki/Bessel_Function_of_the_First_Kind_for_Imaginary_Argument
    #return jjv_val


import jax
import jax.numpy as jnp
import scipy.special
from jax import custom_jvp, pure_callback

# see https://github.com/google/jax/issues/11002


def generate_bessel(function):
    """function is Jv, Yv, Hv_1,Hv_2"""

    @custom_jvp
    def cv(v, x):
        return pure_callback(
            lambda vx: function(*vx),
            x,
            (v, x),
            vectorized=True,
        )

    @cv.defjvp
    def cv_jvp(primals, tangents):
        v, x = primals
        dv, dx = tangents
        primal_out = cv(v, x)

        ## https://dlmf.nist.gov/10.6 formula 10.6.1
        #tangents_out = jax.lax.cond(
            #v == 0,
            #lambda: -cv(v + 1, x),
            #lambda: 0.5 * (cv(v - 1, x) - cv(v + 1, x)),
        #)
        
        tcv_0 = -cv(v + 1, x)
        tcv_p =  0.5 * (cv(v - 1, x) - cv(v + 1, x))
        
        where = v==0
        
        tangents_out = where*tcv_0 + (1-where)*tcv_p
                

        return primal_out, tangents_out * dx

    return cv


jv = generate_bessel(scipy.special.jv)



@custom_jvp
def fresnel_sin(z):
    return pure_callback(sp_fresnel_sin, jax.ShapeDtypeStruct(z.shape, np.float64),vectorized=True)

@fresnel_sin.defjvp
def fresnel_sin_jvp(primal,tangent):
    # https://math.byu.edu/~bakker/Math113/Lectures/M113Lec01.pdf
    z = primal[0]
    primal_out = fresnel_sin(z)

    tangent_out = jnp.sin((jnp.pi*z**2)/2)*tangent[0]

    return primal_out, tangent_out


@custom_jvp
def fresnel_cos(z):
    return pure_callback(sp_fresnel_cos, jax.ShapeDtypeStruct(z.shape, np.float64),vectorized=True)

@fresnel_cos.defjvp
def fresnel_cos_jvp(primal,tangent):
    # https://math.byu.edu/~bakker/Math113/Lectures/M113Lec01.pdf
    z = primal[0]
    primal_out = fresnel_cos(z)

    tangent_out = jnp.cos((jnp.pi*z**2)/2)*tangent[0]

    return primal_out, tangent_out

def sp_fresnel_sin(z): return sp.fresnel(z)[0]
def sp_fresnel_cos(z): return sp.fresnel(z)[1]


## ----------------------------------------------------------------------------------------------------------------------
##  Host Fun Functions
## ----------------------------------------------------------------------------------------------------------------------

#def host_fresnel_sin(z: np.ndarray) -> np.ndarray:

    #arr = sp.fresnel(z)

    #return arr[0]


#def host_fresnel_sin_auto(m: jnp.ndarray) -> jnp.ndarray:

    #z = tf.Variable(z)
    
    #with tf.GradientTape() as tape:
        #y = fresnel_sin(z)
    
    #dy_dz = tape.gradient(y,z)   

    #return dy_dz.numpy()


#def call_jax_other_device_FS(arg):
    #"""Calls a JAX function on a specific device with simple support for reverse AD.
    #Functions whose name starts with "jax_outside" are called on another device,
    #by way of hcb.call.
    #"""

    #@api.custom_vjp
    #def make_call(arg):
        #return hcb.call(host_fresnel_sin, arg,
                        #result_shape=jax.ShapeDtypeStruct(arg.shape, np.float64))

    ## Define the fwd and bwd custom_vjp functions
    #def make_call_vjp_fwd(arg):
        ## Return the primal argument as the residual. Use `make_call` for the
        ## primal computation to enable higher-order AD.
        #return make_call(arg), arg  # Return the primal argument as the residual

    #def make_call_vjp_bwd(res, ct_res):
        #arg = res  # residual is the primal argument

        #def jax_outside_vjp_fun(arg_and_ct):
            #arg, ct = arg_and_ct
            #_, f_vjp = api.vjp(host_fresnel_sin_auto, arg)
            #ct_in, = f_vjp(ct)
            #return ct_in

        #return (call_jax_other_device_FS(jax_outside_vjp_fun, (arg, ct_res)),)

    #make_call.defvjp(make_call_vjp_fwd, make_call_vjp_bwd)
    #return make_call(arg)