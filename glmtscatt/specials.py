# -*- coding: utf-8 -*-
"""
Module for declaring special functions to be used in GLMT.

@author: Luiz Felipe Machado Votto
"""

from scipy import misc
from scipy import special
import numpy as np

def memoize(function):
    CACHE = {}
    def memoized_function(*args):
        if args in CACHE:
            return CACHE[(args)]
        else:
            return function(*args)
    return memoized_function

def squared_bessel_0(argument, scale):
    """ Squared value for order 0 regular Bessel function """
    return pow(special.j0(scale * argument), 2)

def _riccati_bessel_j(degree, argument):
    """ Riccati-Bessel function of first kind and derivative """
    return special.riccati_jn(degree, float(argument))

def _riccati_bessel_y(degree, argument):
    """ Riccati-Bessel function of first kind and derivative """
    return (np.array(special.riccati_jn(degree, float(argument))) 
            - 1j * np.array(special.riccati_yn(degree, float(argument))))

def riccati_bessel_j(degree, argument):
    """ Riccati-Bessel function of first kind """
    return special.riccati_jn(degree, float(argument))[0][degree]

def riccati_bessel_y(degree, argument):
    """ Riccati-Bessel function of second kind """
    return special.riccati_jn(degree, float(argument))[0][degree] \
           - 1j * special.riccati_yn(degree, float(argument))[0][degree]

def d_riccati_bessel_j(degree, argument):
    """ Derivative of Riccati-Bessel function of first kind """
    return special.riccati_jn(degree, float(argument))[1][degree]

def d_riccati_bessel_y(degree, argument):
    """ Derivative of Riccati-Bessel function of second kind """
    return special.riccati_jn(degree, float(argument))[1][degree] \
           - 1j * special.riccati_yn(degree, float(argument))[1][degree]

def d2_riccati_bessel_j(degree, argument):
    """ Second order derivative of Riccati-Bessel function of first kind
    
    .. math::
        \\Psi_n''(x) = \\frac{(1+n^2-x^2)\\Psi_n^{(1)}(x)}{x},
    
    where :math:`\\Psi_n^{(1)}` is the Spherical Bessel function of first kind.
    """
    if argument == 0:
        argument = 1E-16
    return 1 / (argument) \
            * (degree + pow(degree, 2) - pow(argument, 2)) \
            * special.spherical_jn(degree, float(argument))

def d2_riccati_bessel_y(degree, argument):
    """ Second order derivative of Riccati-Bessel function of second kind
    
    .. math::
        \\xi_n''(x) = \\frac{(1+n^2-x^2)\\Psi_n^{(4)}(x)}{x},
    
    where :math:`\\Psi_n^{(4)}` is the Spherical Bessel function of fourth kind.
    """
    if argument == 0:
        argument = 1E-16
    return 1 / (argument) \
            * (degree + pow(degree, 2) - pow(argument, 2)) \
            * (special.spherical_jn(degree, float(argument)) \
               - 1j * special.spherical_yn(degree, float(argument)))

def riccati_bessel_radial_i(degree, argument):
    return d2_riccati_bessel_j(degree, argument) \
           + riccati_bessel_j(degree, argument)
        
        
def riccati_bessel_radial_s(degree, argument):
    return d2_riccati_bessel_y(degree, argument) \
           + riccati_bessel_y(degree, argument)

def fac_plus_minus(n, m):
    """ Calculates the expression below avoiding overflows.
    
    .. math::
        \\frac{(n + m)!}{(n - m)!}
    """
    return special.gammasgn(n) * special.gammasgn(m) \
        * np.exp(special.loggamma(n) - special.loggamma(m))

def legendre_p(degree, order, argument):
    """ Associated Legendre function of integer order
    """
    if degree < np.abs(order):
        return 0
    if order < 0:
        return pow(-1, -order) / fac_plus_minus(degree, -order) \
               * legendre_p(degree, -order, argument)
    return special.lpmv(order, degree, argument)

def legendre_tau(degree, order, argument, mv=True):
    """ Returns generalized Legendre function tau

    Derivative is calculated based on relation 14.10.5:
    http://dlmf.nist.gov/14.10
    """
    if not mv:
        return -np.sqrt(1 - argument ** 2) * special.lpmn(order, degree, argument)[1][order][degree]
    '''
    if argument == 1:
        argument = 1 - 1E-16
    if argument == -1:
        argument = -1 + 1E-16
    '''
    return (degree * argument * legendre_p(degree, order, argument) \
            - (degree + order) * legendre_p(degree - 1, order, argument)) \
            / (np.sqrt(1 - argument * argument))

def legendre_pi(degree, order, argument, overflow_protection=False):
    """ Generalized associated Legendre function pi
    
    .. math::
        \\pi_n^m(x) = \\frac{P_n^m(x)}{\\sqrt{1-x^2}}
    """
    if overflow_protection:
        if argument == 1:
            argument = 1 - 1E-16
        if argument == -1:
            argument = -1 + 1E-16
    return legendre_p(degree, order, argument) \
           / (np.sqrt(1 - argument * argument))

