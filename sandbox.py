# -*- coding: utf-8 -*-
"""
Sandbox for better understanding of the application's structure. In further
commits, this file should not exist and its parts should become auxiliary
modules for the application.

Author: Luiz Felipe Machado Votto
"""

import time
from scipy import special
import numpy as np
import matplotlib.pyplot as plt

from field import SphericalField, CartesianField
from specials import (_riccati_bessel_j, d2_riccati_bessel_j,
                      legendre_p, legendre_tau, legendre_pi)

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


def protected_denominator(value, epsilon=np.longdouble(1E-25)):
    """ Changes a value that's zero to a small number. """
    if value == 0:
        return epsilon
    else:
        return value

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
        s.append(function(j * 1E-6, *args))
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

def zero(*args, **kwargs):
    """ The zero constant function """
    return 0

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
    
def test_fields():
    electric_i_tm = SphericalField(radial=radial_electric_i_tm,
                                   theta=theta_electric_i_tm,
                                   phi=phi_electric_i_tm)
    
    electric_i_te = SphericalField(radial=zero, theta=theta_electric_i_te,
                                   phi=phi_electric_i_tm)
    
    print('TEST: ', electric_i_tm.evaluate(radial=1E-6, theta=np.pi/2, phi=0,
                                 wave_number_k=WAVE_NUMBER))
    print('REF: ', radial_electric_i_tm(1E-6, np.pi/2, 0, WAVE_NUMBER),
                   theta_electric_i_tm(1E-6, np.pi/2, 0, WAVE_NUMBER),
                   phi_electric_i_tm(1E-6, np.pi/2, 0, WAVE_NUMBER))
    
    print('TEST: ', electric_i_te.evaluate(radial=1E-6, theta=np.pi/2, phi=0,
                                 wave_number_k=WAVE_NUMBER))
    print('REF: ', 0,
                   theta_electric_i_te(1E-6, np.pi/2, 0, WAVE_NUMBER),
                   phi_electric_i_te(1E-6, np.pi/2, 0, WAVE_NUMBER))
    
    print(' ')
    print((electric_i_te + electric_i_tm).evaluate(radial=1E-6, theta=np.pi/2, phi=0,
                                 wave_number_k=WAVE_NUMBER)[1])
    print(theta_electric_i_te(1E-6, np.pi/2, 0, WAVE_NUMBER) + theta_electric_i_tm(1E-6, np.pi/2, 0, WAVE_NUMBER))

def abs_func(func):
    def absolute_func(*args, **kwargs):
        return abs(func(*args, **kwargs))
    return absolute_func

def square_func(func):
    def squared_func(*args, **kwargs):
        return pow(func(*args, **kwargs), 2)
    return squared_func

def test_cartesian_fields():
    electric_i_tm = SphericalField(radial=radial_electric_i_tm,
                                   theta=theta_electric_i_tm,
                                   phi=phi_electric_i_tm)
    electric_i_te = SphericalField(radial=zero, theta=theta_electric_i_te,
                                   phi=phi_electric_i_tm)
    electric_i = electric_i_te + electric_i_tm
    
    cartesian_electric_i = CartesianField(spherical=electric_i)
    """
    print(cartesian_electric_i.functions['x'](1E-10, 1E-100, 1E-100, WAVE_NUMBER))
    print(electric_i_tm.functions['radial'](1E-10, np.pi/2, 0, WAVE_NUMBER))
    print(electric_i.functions['radial'](1E-10, np.pi/2, 0, WAVE_NUMBER))
    print(cartesian_electric_i.abs(x=1E-10,y=1E-10,z=1E-10, wave_number_k=WAVE_NUMBER))
    print(cartesian_electric_i.evaluate(x=1E-10,y=1E-100,z=1E-10, wave_number_k=WAVE_NUMBER))
    """
    start_time = time.time()
    do_some_plotting(square_absolute_electric_i, np.pi/2, 0, WAVE_NUMBER)
    print("::: SPHERICAL :::")
    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    t = np.linspace(START, STOP / 1E-6, num=NUM)
    s = []
    for j in t:
        s.append(pow(cartesian_electric_i.abs(x=j * 1E-6,y=1E-100,z=1E-100, wave_number_k=WAVE_NUMBER), 2))
    plt.plot(t, s)
    print("::: CARTESIAN :::")
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.show()

def test_meshgrid():
    pass

def test_plot_2d_bessel():     
    fig, axs = plt.subplots()
    electric_i_tm = SphericalField(radial=radial_electric_i_tm,
                                   theta=theta_electric_i_tm,
                                   phi=phi_electric_i_tm)
    electric_i_te = SphericalField(radial=zero, theta=theta_electric_i_te,
                                   phi=phi_electric_i_tm)
    electric_i = electric_i_te + electric_i_tm
    
    cartesian_electric_i = CartesianField(spherical=electric_i)
    
    x = np.linspace(-STOP/1E-6, STOP/1E-6, 100)
    X, Y = np.meshgrid(x, x)

    start_time = time.time()
    zdata = np.vectorize(cartesian_electric_i.abs)(x=1E-6*X, y=1E-6*Y, z=0, wave_number_k=WAVE_NUMBER)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    levels = np.linspace(0, 1, 40)
    
    cs = axs.contourf(X, Y, zdata, levels=levels)
    fig.colorbar(cs, ax=axs, format="%.2f")
    
    plt.show()


MAX_IT = get_max_it(STOP)

#test_h_field_vs_bessel()
#test_e_field_vs_bessel()
#test_radial_convergence()
#test_cartesian_fields()

test_plot_2d_bessel()

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

import winsound
Freq = 2500 # Set Frequency To 2500 Hertz
Dur = 1000 # Set Duration To 1000 ms == 1 second
winsound.Beep(Freq,Dur)