# -*- coding: utf-8 -*-
"""
Sandbox for better understanding of the application's structure. In further
commits, this file should not exist and its parts should become auxiliary
modules for the application.

Author: Luiz Felipe Machado Votto
"""

from abc import ABC, abstractmethod
from scipy import misc
from scipy import special
import numpy as np
import matplotlib.pyplot as plt


EPSILON = 0
AXICON = np.pi / 180  # 1 degree
# 0.349066  # 20 degrees
MAX_IT = 200
WAVELENGTH = 1064E-9
WAVE_NUMBER = 2 * np.pi / WAVELENGTH


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
             
def protected_denominator(value, epsilon=1E-100):
    if value == 0:
        return epsilon
    else:
        return value

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
    if not isinstance(degree, int):
        print(degree)
    return 1 / protected_denominator(argument) \
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
    return (degree * argument * legendre_p(degree, order, argument) \
            - (degree + order) * legendre_p(degree - 1, order, argument)) \
            / protected_denominator(np.sqrt(1 - argument * argument))

def legendre_pi(degree, order, argument):
    """ Generalized associated Legendre function pi
    """
    return legendre_p(degree, order, argument) \
           / protected_denominator(np.sqrt(1 - argument * argument))

def beam_shape_g_exa(degree, axicon=AXICON, bessel=True):
    """ Computes exact BSC from equations referenced by the article
    """
    if bessel:
        return special.j0((degree + 1/2) * np.sin(axicon))
    return (1 + np.cos(axicon)) / (2 * degree * (degree + 1)) \
            * (legendre_tau(degree, 1, np.cos(axicon)) \
               + legendre_pi(degree, 1, np.cos(axicon)))
    

def beam_shape_g(degree, order, axicon=AXICON, mode='TM'):
    """ Computes BSC in terms of degree and order
    """
    if mode == 'TM':
        return beam_shape_g_exa(degree, axicon=axicon) / 2
    if mode == 'TE':
        retval = 1j * beam_shape_g_exa(degree, axicon=axicon) / 2
        if order > 0:
            return retval
        else:
            return -retval

def plane_wave_coefficient(degree, wave_number_k):
    """ Computes plane wave coefficient c_{n}^{pw}
    """
    return (1 / (1j * wave_number_k)) \
            * pow(-1j, degree) \
            * (2 * degree + 1) / (degree * (degree + 1))

def radial_electric_i_tm(radial, theta, phi, wave_number_k):
    """ Computes the radial component of inciding electric field in TM mode.
    """
    result = 0
    n = 1
    increment = EPSILON + 1

    riccati_bessel_list = _riccati_bessel_j(MAX_IT, wave_number_k * radial)
    riccati_bessel = riccati_bessel_list[0]

    while n < MAX_IT and abs(increment) >= EPSILON:
        for m in [-1, 1]:
            increment = plane_wave_coefficient(n, wave_number_k) \
                      * beam_shape_g(n, m, mode='TM') \
                      * (d2_riccati_bessel_j(n, wave_number_k * radial) \
                         + riccati_bessel[n]) \
                      * legendre_p(n, abs(m), np.cos(theta)) \
                      * np.exp(1j * m * phi)
            result += increment
        n += 1
        
    return wave_number_k * result
        
def theta_electric_i_tm(radial, theta, phi, wave_number_k):
    result = 0
    n = 1
    increment = EPSILON + 1

    riccati_bessel_list = _riccati_bessel_j(MAX_IT, wave_number_k * radial)
    d_riccati_bessel = riccati_bessel_list[1]

    while n < MAX_IT and abs(increment) > EPSILON:
        for m in [-1, 1]:
            increment = plane_wave_coefficient(n, wave_number_k) \
                      * beam_shape_g(n, m, mode='TM') \
                      * d_riccati_bessel[n] \
                      * legendre_tau(n, abs(m), np.cos(theta)) \
                      * np.exp(1j * m * phi)
            result += increment
        n += 1
        
    return result / protected_denominator(radial)

def theta_electric_i_te(radial, theta, phi, wave_number_k):
    result = 0
    n = 1
    increment = EPSILON + 1

    riccati_bessel_list = _riccati_bessel_j(MAX_IT, wave_number_k * radial)
    riccati_bessel = riccati_bessel_list[0]

    while n < MAX_IT and abs(increment) > EPSILON:
        for m in [-1, 1]:
            increment = m \
                      * plane_wave_coefficient(n, wave_number_k) \
                      * beam_shape_g(n, m, mode='TE') \
                      * riccati_bessel[n] \
                      * legendre_pi(n, abs(m), np.cos(theta)) \
                      * np.exp(1j * m * phi)
            result += increment
        n += 1
        
    return result / protected_denominator(radial)

def phi_electric_i_tm(radial, theta, phi, wave_number_k):
    result = 0
    n = 1
    increment = EPSILON + 1

    riccati_bessel_list = _riccati_bessel_j(MAX_IT, wave_number_k * radial)
    d_riccati_bessel = riccati_bessel_list[1]

    while n < MAX_IT and abs(increment) > EPSILON:
        for m in [-1, 1]:
            increment = m \
                      * plane_wave_coefficient(n, wave_number_k) \
                      * beam_shape_g(n, m, mode='TM') \
                      * d_riccati_bessel[n] \
                      * legendre_pi(n, abs(m), np.cos(theta)) \
                      * np.exp(1j * m * phi)
            result += increment
        n += 1
        
    return 1j * result / protected_denominator(radial)

def phi_electric_i_te(radial, theta, phi, wave_number_k):
    result = 0
    n = 1
    m = 0
    increment = EPSILON + 1

    riccati_bessel_list = _riccati_bessel_j(MAX_IT, wave_number_k * radial)
    riccati_bessel = riccati_bessel_list[0]

    while n < MAX_IT and abs(increment) > EPSILON:
        for m in [-1, 1]:
            increment = plane_wave_coefficient(n, wave_number_k) \
                      * beam_shape_g(n, m, mode='TE') \
                      * riccati_bessel[n] \
                      * legendre_tau(n, abs(m), np.cos(theta)) \
                      * np.exp(1j * m * phi)
            result += increment
        n += 1
        
    return 1j * result / protected_denominator(radial)

def abs_theta_electric_i(radial, theta, phi, wave_number_k):
    retval = theta_electric_i_tm(radial, theta, phi, wave_number_k)
    retval += theta_electric_i_te(radial, theta, phi, wave_number_k)
    return abs(retval)

def abs_phi_electric_i(radial, theta, phi, wave_number_k):
    retval = phi_electric_i_tm(radial, theta, phi, wave_number_k)
    retval += phi_electric_i_te(radial, theta, phi, wave_number_k)
    return abs(retval)

def square_absolute(radial, theta, phi, wave_number_k):
    retval = pow(abs(radial_electric_i_tm(radial, theta, phi, wave_number_k)), 2)
    retval += pow(abs_theta_electric_i(radial, theta, phi, wave_number_k), 2)
    retval += pow(abs_phi_electric_i(radial, theta, phi, wave_number_k),2)
    return retval

START = EPSILON
#STOP = 10/(WAVE_NUMBER * np.sin(AXICON))
STOP = 10/(WAVE_NUMBER) # (to see the peak)
NUM = 500

def normalize_list(lst):
    try:
        maxlst = max(lst)
        is_parent = False
    except ValueError:
        is_parent = True
        for sublist in lst:
            normalize_list(sublist)
    if not is_parent:
        for i in range(0,len(lst)):
            lst[i] /= maxlst

def do_some_plotting(function, *args, start=START, stop=STOP, num=NUM, normalize=False):
        t = np.linspace(start, stop, num=num)
        s=[]
        for j in t:
            s.append(function(j, *args))
        if normalize:
            normalize_list(s)
        maxs = max(s)
        print('PEAK = ', maxs, ' at ', t[s.index(maxs)])
        plt.plot(t, s)


def bessel_0(argument, scale):
    return pow(special.j0(scale * argument), 2)

def get_max_it(x_max, wave_number_k=WAVE_NUMBER):
    """ Calculates stop iteration number """
    return np.ceil(wave_number_k * x_max + 4.05 \
                   * pow(wave_number_k * x_max, 1/3)) + 2
    
MAX_IT = get_max_it(STOP)
print('MAX_IT = ', MAX_IT)
#for MAX_IT in range(2,20):
do_some_plotting(bessel_0, WAVE_NUMBER*np.sin(AXICON))
do_some_plotting(square_absolute, -np.pi/2, 0, WAVE_NUMBER)

plt.show()

# do_some_plotting(radial_electric_i_tm, np.pi/2, 0, WAVE_NUMBER)

        
    