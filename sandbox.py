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

EPSILON = 1E-5
AXICON = 0.349066  # 20 degrees
MAX_IT = 1E4

def bromwich_scalar_g_exa(degree, axicon=AXICON):
    """ Computes exact BSC

    From eq. 5 in AMBROSIO, Leonardo et al. !!! NEED REFERENCE
    """
    # FIXME
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

    From eq. (III.3) in GOUESBET, Gerárd !!! NEED REFERENCE
    """
    # FIXME
    return (1 / (np.complex(0, 1) * wave_number_k)) \
            * pow(-np.complex(0, 1), degree) \
            * (2 * degree + 1) / (degree * (degree + 1))

def radial_electric_i_tm(radial, theta, phi, wave_number_k):
    """ Computes the radial component of inciding electric field in TM mode.
    """
    error = float('inf')
    result = 0
    last_result = 0
    n = 0
    m = 0

    while error > EPSILON and n < MAX_IT:
        for m in [-1, 1]:
            result += plane_wave_coefficient(n, wave_number_k) \
                      * bromwich_scalar_g(n, m) \
                      * (d2_riccati_bessel_j(n, wave_number_k * radial)[n] \
                       - riccati_bessel_j(n, wave_number_k * radial)[n]) \
                      * legendre_p(n, m, np.cos(theta)) \
                      * np.exp(np.imag * m * phi)
        error = np.abs(last_result - result).sum()
        last_result = result[:]
        n += 1
