# -*- coding: utf-8 -*-
"""
Sandbox for better understanding of the application's structure. In further
commits, this file should not exist and its parts should become auxiliary
modules for the application.

Author: Luiz Felipe Machado Votto
"""

from abc import ABC, abstractmethod
from scipy import special
import numpy as np


EPSILON = 1E-5
AXICON = 0.349066  # 20 degrees
MAX_IT = 1E4


class Field(ABC):
    """ This is representative of a Field in tridimensional space.
    """
    functions = {}

    def __init__(self, **kwargs):
        self.functions = kwargs

    def evaluate(self, *args, **kwargs):
        """ Evaluates the value of the field given a point.
        """
        result = []
        for key in kwargs:
            result.append(self.functions[key](*args, **kwargs) or 0)
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
        return np.abs(self.functions['r'](radial, theta, phi))


class CartesianField(Field):
    """ Represents a tridimensional field in cartesian coordinates
    """
    def __init__(self, x=None, y=None, z=None):
        kwargs = {'x' : x, 'y' : y, 'z' : z}
        super(CartesianField, self).__init__(**kwargs)

    def abs(self, x_value, y_value, z_value):
        return np.linalg.norm(
            self.evaluate(x=x_value, y=y_value, z=z_value))

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
    """ Second order derivative of Riccati-Bessel function of first kind

    This was made based on relations 10.51.2 from: http://dlmf.nist.gov/10.51
    """
    values = _riccati_bessel_j(degree, argument)
    values_2 = np.roll(values[0], 2)
    values_2[0] = values_2[1] = 0
    print('VALUES = ', values)
    print('VALUES 0 = ', values[0])
    print('VALUES 1 = ', values[1])
    print('VALUES 2 = ', values_2)

    return values_2 \
           - (degree + 1) * degree \
             * values[0] / (argument * argument) \
           - 2 * (degree + 1) * values[1] / argument

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

def bromwich_scalar_g_exa(degree, axicon=AXICON):
    """ Computes exact BSC from equations referenced by the article
    """
    return (1 + np.cos(axicon)) / (2 * degree * (degree + 1)) \
            * (legendre_tau(degree, 1, np.cos(axicon)) \
               + legendre_pi(degree, 1, np.cos(axicon)))

def bromwich_scalar_g(degree, order, axicon=AXICON, mode='TM'):
    """ Computes BSC in terms of degree and order
    """
    if mode == 'TM':
        return bromwich_scalar_g_exa(degree, axicon=axicon) / 2
    if mode == 'TE':
        retval = np.complex(0, 1) \
                     * bromwich_scalar_g_exa(degree, axicon=axicon) / 2
        if order > 0:
            return retval
        else:
            return -retval

def plane_wave_coefficient(degree, wave_number_k):
    """ Computes plane wave coefficient c_{n}^{pw}
    """
    return (1 / (np.complex(0, 1) * wave_number_k)) \
            * pow(-np.complex(0, 1), degree) \
            * (2 * degree + 1) / (degree * (degree + 1))

def radial_electric_i_tm(radial, theta, phi, wave_number_k):
    """ Computes the radial component of inciding electric field in TM mode.
    """
    error = float('inf')
    result = 0
    last_result = 0
    n = 1
    m = 0

    riccati_bessel_list = _riccati_bessel_j(MAX_IT, wave_number_k * radial)
    riccati_bessel = riccati_bessel_list[0]

    d2_riccati_bessel = d2_riccati_bessel_j(MAX_IT, wave_number_k * radial)

    while error > EPSILON and n < MAX_IT:
        for m in [-1, 1]:
            result += plane_wave_coefficient(n, wave_number_k) \
                      * bromwich_scalar_g(n, m, mode='TM') \
                      * (d2_riccati_bessel[n] - riccati_bessel[n]) \
                      * legendre_p(n, m, np.cos(theta)) \
                      * np.exp(np.complex(0, 1) * m * phi)
        error = np.abs(last_result - result).sum()
        last_result = result
        n += 1

    print('IT =', n)

    return result
