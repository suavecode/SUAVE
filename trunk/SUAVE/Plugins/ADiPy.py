"""
ADiPy: Automatic Differentiation for Python

A simple tool for handling arbitrary order automatic
differentiation (AD).

See the examples at the bottom for some simple and advanced uses.

NOTES:

1. Arbitrary order differentiation only supported for univariate calculations
2. 1st-order differentiation supported for multivariate calculations (use
   numpy arrays).
3. Jacobian matrix generator
4. Univariate Taylor Series function generator
   
Copyright 2013: Abraham Lee

"""

from __future__ import division
import autograd.numpy as np 
from math import factorial
import copy

class ad(object):

    def __init__(self, nom=None, der=None):
        """
        AD class constructor. A single nominal or a 1-d array of n nominal 
        values supported. If no value given for "der", then the default
        is "1" for a single nominal or an n-by-n identity matrix for a
        nominal array. Constants can be used by setting "der=0".
        """
        if nom is None:
            self.val = []
            self.der = []
        elif der is None:  # for constant with derivative 0
            self.val = nom
            try:
                nom[0]
            except:
                self.der = 1
            else:
                self.der = np.eye(len(nom))
        else:
            self.val = nom
            self.der = der
    
    def copy(self):
        return copy.copy(self)
        
    @property
    def nom(self):
        """
        Get the 0th-order value (i.e., the nominal value
        """
        return taylornominal(self)
    
    def d(self, n):
        """
        Get the nth derivative
        
        Example
        -------
        A 3rd-order differentiable object at 1.5::
        
            >>> x = adn(1.5, 3)
            >>> y = x**2
            >>> y.d(1)
            3.0
            >>> y.d(2)
            2.0
            >>> y.d(3)
            0.0
            
        """
        assert n>=0, 'Derivative order must not be negative.'
        if n==0:
            return self.nom
        else:
            derivs = taylorderivatives(self)
            assert len(derivs)>=n, \
                "You didn't track derivatives of order = {}".format(n)
            return derivs[n - 1]
    
    def __getitem__(self, idx):
        return ad(self.val[idx], self.der[idx])
    
    def __len__(self):
        return len(self.val)
        
    def __repr__(self):
        return 'ad' + repr((self.val, self.der))
    
    def __str__(self):
        return 'ad' + repr((self.nom, taylorderivatives(self)))
        
    def double(self):
        """
        Convert ad object to vector of doubles
        """
        return np.hstack((self.val*1.0, self.der*1.0))
        
    def __add__(self, other):
        if isinstance(other, ad):
            return ad(self.val + other.val, self.der + other.der)
        else:
            return ad(self.val + other, self.der)
    
    def __radd__(self, other):
        return self + other
        
    def __neg__(self):
        return -1*self
        
    def __sub__(self, other):
        return self + (-1)*other
    
    def __rsub__(self, other):
        return (-1)*self + other
    
    def __mul__(self, other):
        if isinstance(other, ad):
            return ad(self.val*other.val, 
                self.val*other.der + self.der*other.val)
        else:
            return ad(self.val*other, self.der*other)
    
    def __rmul__(self, other):
        return self*other
    
    def __div__(self, other):
        return self*(other**-1)
    
    def __rdiv__(self, other):
        return other*(self**-1)
    
    def __truediv__(self, other):
        return self.__div__(other)
    
    def __rtruediv__(self, other):
        return self.__rdiv__(other)
    
    def __pow__(self, other):
        if isinstance(other, ad):
            return ad(self.val**other.val, 
                other.val*self.val**(other.val - 1)*self.der + \
                self.val**other.val*log(self.val)*other.der)
        else:
            return ad(self.val**other, other*self.val**(other - 1)*self.der)
    
    def __rpow__(self, other):
        return ad(other**self.val, other**self.val*log(other)*self.der)
    
    def __abs__(self):
        if self==0:
            return ad(0*self.val, 0*self.der)
        else:
            return (self**2)**0.5
    
    def __eq__(self, other):
        if isinstance(other, ad):
            return self.nom==other.nom
        else:
            return self.nom==other
        
    def __lt__(self, other):
        if isinstance(other, ad):
            return self.nom<other.nom
        else:
            return self.nom<other
        
    def __le__(self, other):
        return self<other or self==other
    
    def __gt__(self, other):
        return not self<=other
    
    def __ge__(self, other):
        return not self<other
    
    def __ne__(self, other):
        return not self==other
    
    def __nonzero__(self):
        return self!=0
        
def adn(nom, order=1):
    """
    Construct an ad object that tracks derivatives up to ``order``.
    
    Parameters
    ----------
    nom : scalar
        The base, nominal value where the derivatives will be calculated at.
    
    Optional
    --------
    order : int
        The greatest order of derivatives to be tracked (Default: 1)
    """
    if order==1:
        return ad(nom)
    else:
        return ad(adn(nom, order - 1))

def taylornominal(u):
    """
    Collect the zeroth order term that would be used in a Taylor polynomial
    expansion (assumes any necessary calculations have been done prior).
    """
    if isinstance(u, ad):
        return taylornominal(u.val)
    else:
        return u

def taylorderivatives(u):
    """
    Collect all the derivative coefficients (unscaled) necessary for a Taylor
    polynomial expansion (assumes any necessary calculations have been done
    prior).
    """
    if isinstance(u, ad):
        order = 0
        tmp = u
        while hasattr(tmp, 'val'):
            tmp = tmp.val
            order += 1
        
        c = [0.0]*order
        for idx in xrange(order):
            tmp = u
            for v in xrange(order - idx - 1):
                tmp = tmp.val
            for d in xrange(idx + 1):
                if hasattr(tmp, 'der'):
                    tmp = tmp.der
                else:
                    break
            c[idx] = tmp
        return np.array(c)
    else:
        return np.ones_like(u)
    
def taylorcoef(u):
    """
    Collect all the scaled derivative coefficients necessary for a Taylor
    polynomial expansion (assumes any necessary calculations have been done
    prior).
    """
    if isinstance(u, ad):
        c = taylorderivatives(u)
        for idx in xrange(len(c)):
            c[idx] = c[idx]/factorial(idx + 1)
        return c
    else:
        return np.ones_like(u)

def taylorterms(u):
    """
    Collect all the coefficients necessary for a Taylor polynomial expansion,
    including the zeroth order term (assumes any necessary calculations have
    been done prior).
    """
    if isinstance(u, ad):
        return np.hstack((taylornominal(u), taylorcoef(u)))
    else:
        return np.array([u])

def taylorfunc(u, at=None):
    """
    Construct a univariate taylor polynomial expansion about a reference point.
    
    Parameters
    ----------
    u : an ad object
        The object that contains information about its derivatives
    at : scalar
        The point about which the series will be expanded (default: 0).
    
    Returns
    -------
    func : function
        The taylor series polynomial function that approximates ``u``, 
        expanded about the point ``at``.
    
    Example
    -------
    ::
    
        >>> from ad import series, taylorfunc
        >>> x = adn(3, 6)  # a sixth order derivative tracker, nominal=3
        >>> f = x*sin(x*x)
        >>> func = taylorfunc(f, 3)
        >>> func(3)
        1.2363554557252698
    """
    if at is None:
        at = 0  # assume it's expanded about the origin
        
    c = taylorterms(u)
    def approxfunc(x):
        try:
            x[0]
        except:
            return c[0] + np.sum([c[i]*(x - at)**i for i in xrange(1, len(c))])
        else:
            tmp = [c[0] + np.sum([c[i]*(xi - at)**i 
                for i in xrange(1, len(c))]) for xi in x]
            return np.array(tmp)
        
    return approxfunc
    
def exp(u):
    try:
        u[0]
    except:
        if isinstance(u, ad):
            return ad(exp(u.val), exp(u.val)*u.der)
        else:
            return np.exp(u)
    else:
        return [exp(ui) for ui in u]

def log(u):
    try:
        u[0]
    except:
        if isinstance(u, ad):
            return ad(log(u.val), 1/(u.val)*u.der)
        else:
            return np.log(u)
    else:
        return [log(ui) for ui in u]

def sqrt(u):
    try:
        u[0]
    except:
        if isinstance(u, ad):
            return ad(sqrt(u.val), u.der/(2*sqrt(u.val)))
        else:
            return np.sqrt(u)
    else:
        return [sqrt(ui) for ui in u]

def sin(u):
    try:
        u[0]
    except:
        if isinstance(u, ad):
            return ad(sin(u.val), cos(u.val)*u.der)
        else:
            return np.sin(u)
    else:
        return [sin(ui) for ui in u]

def cos(u):
    try:
        u[0]
    except:
        if isinstance(u, ad):
            return ad(cos(u.val), -sin(u.val)*u.der)
        else:
            return np.cos(u)
    else:
        return [cos(ui) for ui in u]
    
def tan(u):
    try:
        u[0]
    except:
        if isinstance(u, ad):
            return ad(tan(u.val), 1/(cos(u.val)**2)*u.der)
        else:
            return np.tan(u)
    else:
        return [tan(ui) for ui in u]
    
def asin(u):
    try:
        u[0]
    except:
        if isinstance(u, ad):
            return ad(asin(u.val), 1/sqrt(1 - u.val**2)*u.der)
        else:
            return np.arcsin(u)
    else:
        return [asin(ui) for ui in u]

def acos(u):
    try:
        u[0]
    except:
        if isinstance(u, ad):
            return ad(acos(u.val), -1/sqrt(1 - u.val**2)*u.der)
        else:
            return np.arccos(u)
    else:
        return [acos(ui) for ui in u]
    
def atan(u):
    try:
        u[0]
    except:
        if isinstance(u, ad):
            return ad(atan(u.val), 1/(1 + u.val**2)*u.der)
        else:
            return np.arctan(u)
    else:
        return [atan(ui) for ui in u]

def jacobian(deps):
    """
    Construct the Jacobian matrix of a set of AD objects that are dependent
    on other independent AD objects.
    
    Parameters
    ----------
    deps : array
        A list of objects that depend on AD objects.
        
    Returns
    -------
    jac : 2d-array
        A matrix where each row corresponds to each dependent AD object and 
        each column corresponds to each independent AD object.
        
    Example
    -------
    Separate ad objects::
    
        >>> x = ad(-1, np.array([1, 0, 0]))
        >>> y = ad(2.1, np.array([0, 1, 0]))
        >>> z = ad(0.25, np.array([0, 0, 1]))
        >>> u = x*y/z
        >>> v = y - z**x
        >>> w = sin(y**2/x)
        >>> jacobian([u, v, w])
        array([[  8.4       ,  -4.        ,  33.6       ],
               [  5.54517744,   1.        ,  16.        ],
               [  1.31330524,   1.25076689,   0.        ]])
    
    A single 3 ad object array::
    
        >>> x = ad(np.array([-1, 2.1, 0.25]))
        >>> y = [0]*3
        >>> y[0] = x[0]*x[1]/x[2]
        >>> y[1] = x[1] - x[2]**x[0]
        >>> y[2] = sin(x[1]**2/x[0])
        >>> jacobian(y)
        array([[  8.4       ,  -4.        ,  33.6       ],
               [  5.54517744,   1.        ,  16.        ],
               [  1.31330524,   1.25076689,   0.        ]])
    
    You can also break-out the individual components before using them::
    
        >>> x, y, z = ad(np.array([-1, 2.1, 0.25]))
        >>> x
        ad(-1.0, array([ 1.,  0.,  0.]))
        >>> y
        ad(2.1, array([ 0.,  1.,  0.]))
        >>> z
        ad(0.25, array([ 0.,  0.,  1.]))
       
    """
    assert all([isinstance(depi, ad) for depi in deps]), 'All inputs must be dependent on AD objects'
    return np.vstack([depi.d(1) for depi in deps])
    
def unite(ad_objs):
    """
    Unite multivariate AD objects into a single AD object
    
    Example
    -------
    
    ::
    
        >>> x = ad(np.array([-1, 2.1, 0.25]))
        >>> y = [0]*2
        >>> y[0] = x[0]*x[1]/x[2]
        >>> y[1] = x[1] - x[2]**x[0]
        # up to this point, y is a list, not an AD object, so y.d(1) won't work
        
        >>> y = unite(y)

        # now, we can do things like y.d(1)
        >>> y.d(1)
        array([[  8.4       ,  -4.        ,  33.6       ],
               [  5.54517744,   1.        ,  16.        ]])

        >>> jacobian(y)  # the result here is the same when y was a list object
        array([[  8.4       ,  -4.        ,  33.6       ],
               [  5.54517744,   1.        ,  16.        ]])
        
        
    """
    assert all([isinstance(adi, ad) for adi in ad_objs]), 'All inputs must be dependent on AD objects'
    try:
        ad_objs[0]
    except:
        return ad_objs
    else:
        ad_noms = np.array([adi.nom for adi in ad_objs])
        ad_ders = np.array([adi.der for adi in ad_objs])
        return ad(ad_noms, ad_ders)

if __name__=='__main__':
    def fdf(a):
        x = ad(a, 1)
        y = exp(-sqrt(x))*sin(x*log(1 + x**2))
        return y.double()
    
    def newtonfdf(a):
        delta = 1
        while abs(delta)>0.000001:
            fvec = fdf(a)
            delta = fvec[0]/fvec[1]
            a = a - delta
        return a
    
    print '\nnewtonfdf(5) should equal 4.8871:\n', newtonfdf(5)
    
    def fgradf(a0, v0, h0):
        a = ad(a0, np.array([1, 0, 0]))  # angle in degrees
        v = ad(v0, np.array([0, 1, 0]))  # velocity in ft/sec
        h = ad(h0, np.array([0, 0, 1]))  # height in ft
        rad = a*np.pi/180
        tana = tan(rad)
        vhor = (v*cos(rad))**2
        f = (vhor/32)*(tana + sqrt(tana**2 + 64*h/vhor))  # horizontal range
        return f.double()
    
    print '\nfgradf(20, 44, 9) should equal [56.0461, 1.0717, 1.9505, 1.4596]:\n', fgradf(20, 44, 9)
    
    def FJF(A):
        x = ad(A[0], np.array([1, 0, 0]))
        y = ad(A[1], np.array([0, 1, 0]))
        z = ad(A[2], np.array([0, 0, 1]))
        f1 = 3*x - cos(y*z) - 0.5
        f2 = x**2 - 81*(y + 0.1)**2 + sin(z) + 1.06
        f3 = exp(-x*y) + 20*z + (10*np.pi - 3)/3
        values = np.array([f1.double(), f2.double(), f3.double()])
        F = values[:, 0]
        J = values[:, 1:4]
        return (F, J)
        
    def newtonFJF(A):
        delta = 1
        while np.max(np.abs(delta))>0.000001:
            [F, J] = FJF(A)
            delta = np.linalg.solve(J, F)
            A = A - delta
        return A
        
    print '\nnewtonFJF([0.1, 0.1, -0.1]) should equal [0.5000, 0.0000, -0.5236]:\n', newtonFJF(np.array([0.1, 0.1, -0.1]))
    
    x = adn(3, 3)
    f = x*sin(x*x)
    print '\nf.d(3) should equal 495.9280:\n', f.d(3)