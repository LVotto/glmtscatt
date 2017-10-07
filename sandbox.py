# -*- coding: utf-8 -*-
"""
Sandbox for better understanding of the application's structure. In further
commits, this file should not exist and its parts should become auxiliary
modules for the application.

Author: Luiz Felipe Machado Votto
"""

from abc import ABC, abstractmethod
#from decimal import Decimal

from scipy import misc
from scipy import special
import numpy as np
import matplotlib.pyplot as plt


EPSILON = 0
AXICON = np.longdouble(np.pi / 180)  # 1 degree
# np.longdouble(0.349066)  # 20 degrees
MAX_IT = 200
WAVELENGTH = np.longdouble(1064.0E-9)
WAVE_NUMBER = 2 * np.pi / WAVELENGTH
STOP = 10/(WAVE_NUMBER * np.sin(AXICON))
#STOP = 10/(WAVE_NUMBER) # (to see the peak)
START = 1e-10
#START = -STOP
NUM = 500


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

def protected_denominator(value, epsilon=np.longdouble(1E-25)):
    """ Changes a value that's zero to a small number. """
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
    return (degree * argument * legendre_p(degree, order, argument) \
            - (degree + order) * legendre_p(degree - 1, order, argument)) \
            / (np.sqrt(1 - argument * argument))

def legendre_pi(degree, order, argument):
    """ Generalized associated Legendre function pi
    """
    return legendre_p(degree, order, argument) \
           / (np.sqrt(1 - argument * argument))

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
        if order in [-1, 1]:
            return beam_shape_g_exa(degree, axicon=axicon) / 2
        else:
            return 0
    if mode == 'TE':
        retval = 1j * beam_shape_g_exa(degree, axicon=axicon) / 2
        if order == 1:
            return -retval
        elif order == -1:
            return retval
        else:
            return 0

def plane_wave_coefficient(degree, wave_number_k):
    """ Computes plane wave coefficient c_{n}^{pw}
    """
    return (1 / (1j * wave_number_k)) \
            * pow(-1j, degree) \
            * (2 * degree + 1) / (degree * (degree + 1))

def _radial_electric_i_tm(radial, theta, phi, wave_number_k):
    n = 1
    m = 1
    result = 0
    increment = EPSILON
    while n <= MAX_IT and abs(increment) >= EPSILON:
        increment = pow(-1j, (n+1)) \
                    * (2 * n + 1) * beam_shape_g(n, 0, mode='TM') \
                    * special.spherical_jn(n, wave_number_k * radial) \
                    / (wave_number_k * radial) \
                    * legendre_p(n, 0, np.cos(theta))
        result += increment

    increment = EPSILON
    while m <= MAX_IT and increment >= EPSILON:
        n = m
        while n <= MAX_IT and abs(increment) >= EPSILON:
            increment = pow(-1j, (n+1)) \
                        * (2 * n + 1) \
                        * special.spherical_jn(wave_number_k * radial) \
                        / (wave_number_k * radial) \
                        * legendre_p(n, abs(m), np.cos(theta)) \
                        * (beam_shape_g(n, m, mode='TM') \
                           * np.exp(1j * m * phi) \
                           + beam_shape_g(n, -m, mode='TM') \
                           * np.exp(1j * -m * phi))
            result += increment
    return result

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
    """ Computes the theta component of inciding electric field in TM mode.
    """
    result = 0
    n = 1
    increment = EPSILON + 1

    riccati_bessel_list = _riccati_bessel_j(MAX_IT, wave_number_k * radial)
    d_riccati_bessel = riccati_bessel_list[1]

    while n <= MAX_IT and abs(increment) >= EPSILON:
        for m in [-1, 1]:
            increment = plane_wave_coefficient(n, wave_number_k) \
                      * beam_shape_g(n, m, mode='TM') \
                      * d_riccati_bessel[n] \
                      * legendre_tau(n, abs(m), np.cos(theta)) \
                      * np.exp(1j * m * phi)
            result += increment
        #print(np.cos(theta), theta)
        n += 1
    return result / (radial)

def theta_electric_i_te(radial, theta, phi, wave_number_k):
    """ Computes the theta component of inciding electric field in TE mode.
    """
    result = 0
    n = 1
    increment = EPSILON + 1

    riccati_bessel_list = _riccati_bessel_j(MAX_IT, wave_number_k * radial)
    riccati_bessel = riccati_bessel_list[0]

    while n <= MAX_IT and abs(increment) >= EPSILON:
        for m in [-1, 1]:
            increment = m \
                      * plane_wave_coefficient(n, wave_number_k) \
                      * beam_shape_g(n, m, mode='TE') \
                      * riccati_bessel[n] \
                      * legendre_pi(n, abs(m), np.cos(theta)) \
                      * np.exp(1j * m * phi)
            result += increment
            #if abs(result) > 1e-7:
                #print('Undesirable', result,'at', radial, 'n = ', n)
        n += 1

    return result / (radial)

def phi_electric_i_tm(radial, theta, phi, wave_number_k):
    """ Computes the phi component of inciding electric field in TM mode.
    """
    result = 0
    n = 1
    increment = EPSILON + 1

    riccati_bessel_list = _riccati_bessel_j(MAX_IT, wave_number_k * radial)
    d_riccati_bessel = riccati_bessel_list[1]

    while n < MAX_IT and abs(increment) >= EPSILON:
        for m in [-1, 1]:
            increment = m \
                      * plane_wave_coefficient(n, wave_number_k) \
                      * beam_shape_g(n, m, mode='TM') \
                      * d_riccati_bessel[n] \
                      * legendre_pi(n, abs(m), np.cos(theta)) \
                      * np.exp(1j * m * phi)
            result += increment
        n += 1

    return 1j * result / (radial)

def phi_electric_i_te(radial, theta, phi, wave_number_k):
    """ Computes the phi component of inciding electric field in TE mode.
    """
    result = 0
    n = 1
    m = 0
    increment = EPSILON + 1

    riccati_bessel_list = _riccati_bessel_j(MAX_IT, wave_number_k * radial)
    riccati_bessel = riccati_bessel_list[0]

    while n < MAX_IT and abs(increment) >= EPSILON:
        for m in [-1, 1]:
            increment = plane_wave_coefficient(n, wave_number_k) \
                      * beam_shape_g(n, m, mode='TE') \
                      * riccati_bessel[n] \
                      * legendre_tau(n, abs(m), np.cos(theta)) \
                      * np.exp(1j * m * phi)
            result += increment
        n += 1

    return 1j * result / (radial)

def abs_theta_electric_i(radial, theta, phi, wave_number_k):
    """ Calculates absolute value of inciding theta component """
    retval = theta_electric_i_tm(radial, theta, phi, wave_number_k) \
             + theta_electric_i_te(radial, theta, phi, wave_number_k)
    return abs(retval)

def abs_phi_electric_i(radial, theta, phi, wave_number_k):
    """ Calculates absolute value of inciding phi component """
    retval = phi_electric_i_tm(radial, theta, phi, wave_number_k) \
             + phi_electric_i_te(radial, theta, phi, wave_number_k)
    return abs(retval)

def square_absolute_electric_i(radial, theta, phi, wave_number_k):
    """ Calculates absolute value of inciding electric wave """
    retval = pow(abs(radial_electric_i_tm(radial, theta, phi, wave_number_k)), 2)
    retval += pow(abs_theta_electric_i(radial, theta, phi, wave_number_k), 2)
    retval += pow(abs_phi_electric_i(radial, theta, phi, wave_number_k), 2)
    return retval

def theta_magnetic_i_tm(radial, theta, phi, wave_number_k):
    """ Computes the theta component of inciding magnetic field in TM mode.
    """
    result = 0
    n = 1
    increment = EPSILON + 1

    riccati_bessel_list = _riccati_bessel_j(MAX_IT, wave_number_k * radial)
    riccati_bessel = riccati_bessel_list[0]

    while n < MAX_IT and abs(increment) >= EPSILON:
        for m in [-1, 1]:
            increment = m \
                      * plane_wave_coefficient(n, wave_number_k) \
                      * beam_shape_g(n, m, mode='TM') \
                      * riccati_bessel[n] \
                      * legendre_pi(n, abs(m), np.cos(theta)) \
                      * np.exp(1j * m * phi)
            result += increment
        n += 1

    return -result / (radial)

def phi_magnetic_i_tm(radial, theta, phi, wave_number_k):
    """ Computes the phi component of inciding magnetic field in TM mode.
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
                      * riccati_bessel[n] \
                      * legendre_tau(n, abs(m), np.cos(theta)) \
                      * np.exp(1j * m * phi)
            result += increment
        n += 1

    return -1j * result / (radial)

def radial_magnetic_i_te(radial, theta, phi, wave_number_k):
    """ Computes the radial component of inciding magnetic field in TE mode.
    """
    result = 0
    n = 1
    increment = EPSILON + 1

    riccati_bessel_list = _riccati_bessel_j(MAX_IT, wave_number_k * radial)
    riccati_bessel = riccati_bessel_list[0]

    while n < MAX_IT and abs(increment) >= EPSILON:
        for m in [-1, 1]:
            increment = plane_wave_coefficient(n, wave_number_k) \
                      * beam_shape_g(n, m, mode='TE') \
                      * (d2_riccati_bessel_j(n, wave_number_k * radial) \
                         + riccati_bessel[n]) \
                      * legendre_p(n, abs(m), np.cos(theta)) \
                      * np.exp(1j * m * phi)
            result += increment
        n += 1

    return wave_number_k * result

def theta_magnetic_i_te(radial, theta, phi, wave_number_k):
    """ Computes the theta component of inciding magnetic field in TE mode.
    """
    result = 0
    n = 1
    increment = EPSILON + 1

    riccati_bessel_list = _riccati_bessel_j(MAX_IT, wave_number_k * radial)
    d_riccati_bessel = riccati_bessel_list[1]

    while n < MAX_IT and abs(increment) >= EPSILON:
        for m in [-1, 1]:
            increment = plane_wave_coefficient(n, wave_number_k) \
                      * beam_shape_g(n, m, mode='TE') \
                      * d_riccati_bessel[n] \
                      * legendre_tau(n, abs(m), np.cos(theta)) \
                      * np.exp(1j * m * phi)
            result += increment
        n += 1

    return result / (radial)

def phi_magnetic_i_te(radial, theta, phi, wave_number_k):
    """ Computes the phi component of inciding magnetic field in TE mode.
    """
    result = 0
    n = 1
    increment = EPSILON + 1

    riccati_bessel_list = _riccati_bessel_j(MAX_IT, wave_number_k * radial)
    d_riccati_bessel = riccati_bessel_list[1]

    while n < MAX_IT and abs(increment) >= EPSILON:
        for m in [-1, 1]:
            increment = m \
                      * plane_wave_coefficient(n, wave_number_k) \
                      * beam_shape_g(n, m, mode='TE') \
                      * d_riccati_bessel[n] \
                      * legendre_pi(n, abs(m), np.cos(theta)) \
                      * np.exp(1j * m * phi)
            result += increment
        n += 1

    return 1j * result / (radial)

def abs_theta_magnetic_i(radial, theta, phi, wave_number_k):
    """ Calculates absolute value of inciding theta component """
    retval = theta_magnetic_i_tm(radial, theta, phi, wave_number_k) \
             + theta_magnetic_i_te(radial, theta, phi, wave_number_k)
    return abs(retval)

def abs_phi_magnetic_i(radial, theta, phi, wave_number_k):
    """ Calculates absolute value of inciding phi component """
    retval = phi_magnetic_i_tm(radial, theta, phi, wave_number_k) \
             + phi_magnetic_i_te(radial, theta, phi, wave_number_k)
    return abs(retval)

def square_absolute_magnetic_i(radial, theta, phi, wave_number_k):
    """ Calculates absolute value of inciding magnetic wave """
    retval = pow(abs(radial_magnetic_i_te(radial, theta, phi, wave_number_k)), 2)
    retval += pow(abs_theta_magnetic_i(radial, theta, phi, wave_number_k), 2)
    retval += pow(abs_phi_magnetic_i(radial, theta, phi, wave_number_k), 2)
    return retval


def normalize_list(lst):
    try:
        maxlst = max(lst)
        is_parent = False
    except ValueError:
        is_parent = True
        for sublist in lst:
            normalize_list(sublist)
    if not is_parent:
        for i in range(0, len(lst)):
            lst[i] /= maxlst

def do_some_plotting(function, *args, start=START, stop=STOP / 1E-6, num=NUM, normalize=False):
    t = np.linspace(start, stop, num=num)
    s = []
    for j in t:
        s.append(function(j * 1E-6, * args))
    print('DONE:', function.__name__)
    if normalize:
        normalize_list(s)
    maxs = max(s)
    print('PEAK = ', maxs, ' at ', t[s.index(maxs)])
    plt.plot(t, s)

def radial_electric_tm_increment(max_it,
                                 radial,
                                 theta=np.pi/2,
                                 phi=0,
                                 wave_number_k=WAVE_NUMBER):
    result = 0
    riccati_bessel_list = _riccati_bessel_j(max_it, wave_number_k * radial)
    riccati_bessel = riccati_bessel_list[0]
    for n in range(1, max_it):
        for m in [-1, 1]:
            increment = plane_wave_coefficient(n, wave_number_k) \
                      * beam_shape_g(n, m, mode='TM') \
                      * (d2_riccati_bessel_j(n, wave_number_k * radial) \
                         + riccati_bessel[n]) \
                      * legendre_p(n, abs(m), np.cos(theta)) \
                      * np.exp(1j * m * phi)
            result += increment
    return abs(wave_number_k * result)

def bessel_0(argument, scale):
    return pow(special.j0(scale * argument), 2)

def get_max_it(x_max, wave_number_k=WAVE_NUMBER):
    """ Calculates stop iteration number """
    return int(np.ceil(wave_number_k * x_max + np.longdouble(4.05) \
                   * pow(wave_number_k * x_max, 1/3)) + 2)

def difference_x(radial):
    return square_absolute_electric_i(radial, np.pi/2, 0, WAVE_NUMBER) \
           - bessel_0(radial, WAVE_NUMBER * np.sin(AXICON))

MAX_IT = get_max_it(STOP)


def plot_increment(x):
    rng = range(1, get_max_it(x))
    s = []
    for n in rng:
        s.append(radial_electric_tm_increment(n, x))
    plt.plot(n, s[-1], 'ro')
    for n in range(rng[-1], 1000):
        s.append(s[-1])
    plt.plot(range(0,1000), s)

def test_e_field_vs_bessel():
    #MAX_IT = 100
    print('MAX_IT = ', MAX_IT)
    #for MAX_IT in range(2,20):
    do_some_plotting(bessel_0, WAVE_NUMBER*np.sin(AXICON))
    do_some_plotting(square_absolute_electric_i, np.pi/2, 0, WAVE_NUMBER)
    plt.show()

def test_radial_convergence(sample=10, maxx=STOP):
    t = np.linspace(0, maxx, sample)
    for x in t:
        plot_increment(x)
        plt.title(x)
        plt.show()
        
def test_h_field_vs_bessel():
    #MAX_IT = 100
    print('MAX_IT = ', MAX_IT)
    #for MAX_IT in range(2,20):
    do_some_plotting(bessel_0, WAVE_NUMBER*np.sin(AXICON))
    do_some_plotting(square_absolute_magnetic_i, np.pi/2, 0, WAVE_NUMBER)
    plt.show()

test_h_field_vs_bessel()
#test_e_field_vs_bessel()
#test_radial_convergence()


"""
plot_increment(10/(WAVE_NUMBER * np.sin(AXICON)))

rng = np.linspace(0, STOP, 10000)
s1 = []
for x in rng:
    s1.append(_riccati_bessel_j(4, WAVE_NUMBER * x)[0])
plt.plot(rng, s1)
plt.show()

for n in range(1, 10):
    s1 = []
    for x in rng:
        s1.append(_riccati_bessel_j(10, WAVE_NUMBER * x)[1][n])
    plt.plot(rng, s1)
    plt.show()

    s1 = []
for x in rng:
    s1.append(_riccati_bessel_j(300, WAVE_NUMBER * x)[1][1]/x)
plt.plot(rng, s1)
plt.show()
"""

