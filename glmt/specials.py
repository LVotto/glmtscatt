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
    """ Riccati-Bessel function of first kind and derivative
    """
    return special.riccati_jn(degree, argument)

def riccati_bessel_j(degree, argument):
    """ Riccati-Bessel function of first kind
    """
    return special.riccati_jn(degree, argument)[0]

def d_riccati_bessel_j(degree, argument):
    """ Derivative of Riccati-Bessel function of first kind
    """
    return special.riccati_jn(degree, argument)[1]

def d2_riccati_bessel_j(degree, argument):
    """ d2 Psi """
    if argument == 0:
        argument = 1E-16
    return 1 / (argument) \
            * (degree + pow(degree, 2) - pow(argument, 2)) \
            * special.spherical_jn(degree, argument)

def legendre_p(degree, order, argument):
    """ Associated Legendre function of integer order
    """
    if order < 0:
        return pow(-1, -order) * misc.factorial(degree + order) \
               / misc.factorial(degree - order) \
               * legendre_p(degree, -order, argument)
    return special.lpmv(order, degree, argument)

def legendre_tau(degree, order, argument):
    """ Returns generalized Legendre function tau

    Derivative is calculated based on relation 14.10.5:
    http://dlmf.nist.gov/14.10
    """
    if argument == 1:
        argument = 1 - 1E-16
    if argument == -1:
        argument = -1 + 1E-16
    return (degree * argument * legendre_p(degree, order, argument) \
            - (degree + order) * legendre_p(degree - 1, order, argument)) \
            / (np.sqrt(1 - argument * argument))

def legendre_pi(degree, order, argument):
    """ Generalized associated Legendre function pi
    """
    if argument == 1:
        argument = 1 - 1E-16
    if argument == -1:
        argument = -1 + 1E-16
    return legendre_p(degree, order, argument) \
           / (np.sqrt(1 - argument * argument))

