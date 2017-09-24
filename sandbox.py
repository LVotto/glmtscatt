# -*- coding: utf-8 -*-
"""
Spyder Editor

Sandbox for better understanding of the application's structure. In further
commits, this file should not exist and its parts should become auxiliary
modules for the application.

Author: Luiz Felipe Machado Votto
"""

from abc import ABC, abstractmethod
from scipy import special
import numpy as np



class Field(ABC):
    """ This is representative of a Field in tridimensional space.
    """
    functions = {}

    def __init__(self, **kwargs):
        self.functions = kwargs

    def evaluate(self, **kwargs):
        """ Evaluates the value of a field given a point.
        """
        result = []
        for key in kwargs:
            result.append(self.functions[key](kwargs[key]))
        return np.array(result)

    @abstractmethod
    def abs(self, coordinate1, coordinate2, coordinate3):
        """ Evaluates magnitude of field in a given point.
        """
        pass

class SphericalField(Field):
    """ Represents a tridimensional field in spherical coordinates
    """
    def __init__(self, r=None, theta=None, phi=None):
        kwargs = {'r' : r, 'theta' : theta, 'phi' : phi}
        super(SphericalField, self).__init__(**kwargs)

    def abs(self, radial, theta, phi):
        return self.functions['r'](radial, theta, phi)

class CartesianField(Field):
    """ Represents a tridimensional field in cartesian coordinates
    """
    def __init__(self, x=None, y=None, z=None):
        kwargs = {'x' : x, 'y' : y, 'z' : z}
        super(CartesianField, self).__init__(**kwargs)

    def abs(self, x_value, y_value, z_value):
        return np.linalg.norm(
            self.evaluate(x=x_value, y=y_value, z=z_value))


def riccati_bessel_j(degree, argument):
    """ Riccati-Bessel function of first kind
    """
    return special.riccati_jn(degree, argument)[0]

def d_riccati_bessel_j(degree, argument):
    """ Derivative of Riccati-Bessel function of first kind
    """
    return special.riccati_jn(degree, argument)[1]

def d2_riccati_bessel_j(degree, argument):
    """ Second order derivative of Riccati-Bessel function of first kind

    This was made based on relations 10.51.2 from: http://dlmf.nist.gov/10.51
    """
    return riccati_bessel_j(degree - 2, argument) \
           - (degree + 1) * degree \
             * riccati_bessel_j(degree, argument) / (argument * argument) \
           - 2 * (degree + 1) * d_riccati_bessel_j(degree, argument) / argument

def legendre_p(degree, order, argument):
    """ Associated Legendre function of integer order
    """
    return special.lpmv(degree, order, argument)

def legendre_tau(degree, order, argument):
    """ Returns generalized Legendre function tau

    Derivative is calculated based on relation 14.10.5:
    http://dlmf.nist.gov/14.10
    """
    return ((degree - order - 1) * legendre_p(degree - 1, order, argument) \
            - degree * argument * legendre_p(degree, order, argument)) \
            / (1 - argument*argument)

def legendre_pi(degree, order, argument):
    """ Generalized associated Legendre function pi
    """
    return legendre_p(degree, order, argument) / (1 - argument * argument)
