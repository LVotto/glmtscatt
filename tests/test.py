# -*- coding: utf-8 -*-
"""
Effective test module.

@author: Luiz Felipe Machado Votto
"""

import time
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import cm

import glmt.glmt as glmt
from glmt.field import SphericalField, CartesianField
from glmt.specials import squared_bessel_0
from glmt.utils import zero, get_max_it, normalize_list
from glmt.constants import AXICON, WAVE_NUMBER

#STOP = 10/(WAVE_NUMBER * np.sin(AXICON))  # for bessel beam
STOP = 200E-6
#STOP = 10/(WAVE_NUMBER) # (to see the peak)
#START = 0
START = -STOP
NUM = 500

def do_some_plotting(function, *args, start=START, stop=STOP / 1E-6,
                     num=NUM, normalize=False):
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

def plot_increment(x):
    rng = range(1, get_max_it(x))
    s = []
    end = 620
    for n in rng:
        s.append(glmt.radial_electric_tm_increment(n, x))
    plt.plot(n, s[-1], 'ro')
    print('Nmax = ', n)
    error = 0
    index = -1
    while error < 1E-3:
        new_index = index-1
        error = abs(s[index] - s[new_index])
        index = new_index
    plt.plot(rng[index], s[index], 'go')
    print('conv = ', rng[index])
    for n in range(rng[-1], end):
        s.append(s[-1])
    plt.plot(range(0, end), s)

def test_e_field_vs_bessel():
    #MAX_IT = 100
    print('MAX_IT = ', MAX_IT)
    #for MAX_IT in range(2,20):
    do_some_plotting(squared_bessel_0, WAVE_NUMBER*np.sin(AXICON))
    do_some_plotting(glmt.square_absolute_electric_i, np.pi/2, 0, WAVE_NUMBER)
    plt.show()

def test_radial_convergence(sample=10, maxx=STOP):
    t = np.linspace(1E-15, maxx, sample)
    for x in t:
        plot_increment(x)
        plt.title(x)
        plt.show()

def test_h_field_vs_bessel():
    #MAX_IT = 100
    print('MAX_IT = ', MAX_IT)
    #for MAX_IT in range(2,20):
    do_some_plotting(squared_bessel_0, WAVE_NUMBER*np.sin(AXICON))
    do_some_plotting(glmt.square_absolute_magnetic_i, np.pi/2, 0, WAVE_NUMBER)
    plt.show()

def test_fields():
    electric_i_tm = SphericalField(radial=glmt.radial_electric_i_tm,
                                   theta=glmt.theta_electric_i_tm,
                                   phi=glmt.phi_electric_i_tm)

    electric_i_te = SphericalField(radial=zero, theta=glmt.theta_electric_i_te,
                                   phi=glmt.phi_electric_i_tm)

    cart_electric_i_tm = CartesianField(spherical=electric_i_tm)

    print('TEST: ', electric_i_tm.evaluate(radial=1E-6, theta=np.pi/2, phi=0,
                                 wave_number_k=WAVE_NUMBER))
    print('REF: ', glmt.radial_electric_i_tm(1E-6, np.pi/2, 0, WAVE_NUMBER),
                   glmt.theta_electric_i_tm(1E-6, np.pi/2, 0, WAVE_NUMBER),
                   glmt.phi_electric_i_tm(1E-6, np.pi/2, 0, WAVE_NUMBER))

    print('TEST: ', electric_i_te.evaluate(radial=1E-6, theta=np.pi/2, phi=0,
                                 wave_number_k=WAVE_NUMBER))
    print('REF: ', 0,
                   glmt.theta_electric_i_te(1E-6, np.pi/2, 0, WAVE_NUMBER),
                   glmt.phi_electric_i_te(1E-6, np.pi/2, 0, WAVE_NUMBER))

    print(' ')
    print((electric_i_te + electric_i_tm).evaluate(radial=1E-6, theta=np.pi/2, phi=0,
                                 wave_number_k=WAVE_NUMBER)[1])
    print(glmt.theta_electric_i_te(1E-6, np.pi/2, 0, WAVE_NUMBER) + \
          glmt.theta_electric_i_tm(1E-6, np.pi/2, 0, WAVE_NUMBER))

    print('[SPH] x = 1E-6: ', electric_i_tm.abs(radial=1E-6, theta=np.pi/2, phi=0, wave_number_k=WAVE_NUMBER))
    print('[CAR] x = 1E-6: ', cart_electric_i_tm.abs(x=1E-6, y=0, z=0, wave_number_k=WAVE_NUMBER))
    print('[SPH] y = 1E-6: ', electric_i_tm.abs(radial=1E-6, theta=np.pi/2, phi=np.pi/2, wave_number_k=WAVE_NUMBER))
    print('[CAR] y = 1E-6: ', cart_electric_i_tm.abs(x=0, y=1E-6, z=0, wave_number_k=WAVE_NUMBER))
    print('[SPH] z = 1E-6: ', electric_i_tm.abs(radial=1E-6, theta=0, phi=0, wave_number_k=WAVE_NUMBER))
    print('[CAR] z = 1E-6: ', cart_electric_i_tm.abs(x=0, y=0, z=1E-6, wave_number_k=WAVE_NUMBER))
    print('[SPH] z = -1E-6: ', electric_i_tm.abs(radial=1E-6, theta=np.pi, phi=0, wave_number_k=WAVE_NUMBER))
    print('[CAR] z = -1E-6: ', cart_electric_i_tm.abs(x=0, y=0, z=-1E-6, wave_number_k=WAVE_NUMBER))

def abs_func(func):
    def absolute_func(*args, **kwargs):
        return abs(func(*args, **kwargs))
    return absolute_func

def square_func(func):
    def squared_func(*args, **kwargs):
        return pow(func(*args, **kwargs), 2)
    return squared_func

def test_cartesian_fields():
    electric_i_tm = SphericalField(radial=glmt.radial_electric_i_tm,
                                   theta=glmt.theta_electric_i_tm,
                                   phi=glmt.phi_electric_i_tm)
    electric_i_te = SphericalField(radial=zero, theta=glmt.theta_electric_i_te,
                                   phi=glmt.phi_electric_i_te)
    electric_i = electric_i_te + electric_i_tm

    cartesian_electric_i = CartesianField(spherical=electric_i)

    start_time = time.time()
    t = np.linspace(-200, 200, num=NUM/4)
    sx = []
    sy = []
    sz = []
    try:
        with open(str(pathlib.Path('frozen_cartesian2.pickle').absolute()), 'rb') as f:
            print('Loading results: ')
            sx, sy, sz = pickle.load(f)
            plt.plot(t, sx, 'tomato', label='eixo-x')
            plt.plot(t, sy, 'navy', label='eixo-y')
            plt.plot(t, sz, 'firebrick', label='eixo-z')
    except FileNotFoundError:
        print('There are no saved results. Calculating...')

        start_time = time.time()
        for j in t:
            sx.append(pow(abs(cartesian_electric_i.abs(x=j * 1E-6, y=0, z=0, wave_number_k=WAVE_NUMBER)),2))
            print("j = ", j, " -> ", get_max_it(abs(j) * 1E-6), ": ", sx[-1])
        plt.plot(t, sx, 'tomato', label='ex')
        print("::: X :::")
        print("--- %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        for j in t:
            sy.append(pow(abs(cartesian_electric_i.abs(x=0, y=j * 1E-6, z=0, wave_number_k=WAVE_NUMBER)),2))
            print("j = ", j, " -> ", get_max_it(abs(j) * 1E-6), ": ", sy[-1])
        plt.plot(t, sy, 'navy', label='ey')
        print("::: Y :::")
        print("--- %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        for j in t:
            sz.append(pow(abs(cartesian_electric_i.abs(x=0, y=0, z=j * 1E-6, wave_number_k=WAVE_NUMBER)),2))
            print("j = ", j, " -> ", get_max_it(abs(j) * 1E-6), ": ", sz[-1])
        plt.plot(t, sz, 'firebrick', label='ez')
        print("::: Z :::")
        print("--- %s seconds ---" % (time.time() - start_time))
        with open(str(pathlib.Path('../pickles/frozen_cartesian2.pickle').absolute()), 'wb') as f:
            pickle.dump((sx, sy, sz), f)

    plt.ylabel('Magnitude do campo elétrico [V/m]')
    plt.xlabel('x [micrômetros]')
    plt.legend(loc=1)
    plt.show()

def test_meshgrid():
    pass

def test_plot_2d_bessel():
    fig, axs = plt.subplots(1, 2, sharey=True)
    electric_i_tm = SphericalField(radial=glmt.radial_electric_i_tm,
                                   theta=glmt.theta_electric_i_tm,
                                   phi=glmt.phi_electric_i_tm)
    electric_i_te = SphericalField(radial=zero, theta=glmt.theta_electric_i_te,
                                   phi=glmt.phi_electric_i_te)
    electric_i = electric_i_te + electric_i_tm

    cartesian_electric_i = CartesianField(spherical=electric_i)

    x = np.linspace(-100, 100, 100)
    X, Y = np.meshgrid(x, x)

    start_time = time.time()
    try:
        print('Searching for results...')
        with open(str(pathlib.Path('../pickles/frozen2d.pickle').absolute()), 'rb') as f:
            xzdata, xydata = pickle.load(f)
            print('Results where found!')
    except FileNotFoundError:
        print('Results not found. Calculating...')
        xzdata = np.vectorize(cartesian_electric_i.abs)(x=1E-6*X, y=0, z=1E-6*Y, wave_number_k=WAVE_NUMBER)
        xydata = np.vectorize(cartesian_electric_i.abs)(x=1E-6*X, y=1E-6*Y, z=0, wave_number_k=WAVE_NUMBER)
        print("--- %s seconds ---" % (time.time() - start_time))
        with open(str(pathlib.Path('../pickles/frozen2d.pickle').absolute()), 'wb') as f:
            pickle.dump((xydata, xzdata), f)

    levels = np.linspace(0, 1, 40)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    xzdata = np.transpose(np.array(xzdata))
    axs[0].contourf(X, Y, xydata.T, levels=levels)
    cs2 = axs[1].contourf(X, Y, xzdata, levels=levels)
    fig.colorbar(cs2, ax=axs[1], format="%.2f")
    axs[0].set_aspect('equal', 'datalim')
    axs[1].set_aspect('equal','datalim')

    axs[0].set_xlabel('y [micrômetros]')
    axs[1].set_xlabel('z [micrômetros]')
    plt.ylabel('x [micrômetros]')

    plt.show()

def test_plot_2d():
    electric_i_tm = SphericalField(radial=glmt.radial_electric_i_tm,
                                   theta=glmt.theta_electric_i_tm,
                                   phi=glmt.phi_electric_i_tm)
    electric_i_te = SphericalField(radial=zero, theta=glmt.theta_electric_i_te,
                                   phi=glmt.phi_electric_i_te)
    electric_i = electric_i_te + electric_i_tm

    cartesian_electric_i = CartesianField(spherical=electric_i)

    x = np.linspace(-10, 10, 100)
    y = np.linspace(0, 100, 100)
    X, Y = np.meshgrid(x, y)

    start_time = time.time()
    try:
        print('Searching for results')
        with open(str(pathlib.Path('../pickles/gfrozen2d.pickle').absolute()), 'rb') as f:
            xzdata = pickle.load(f)
            print('Results where found!')
    except FileNotFoundError:
        print('Results not found. Calculating...')
        xzdata = np.vectorize(cartesian_electric_i.abs)(x=1E-6*X, y=0, z=1E-6*Y, wave_number_k=WAVE_NUMBER)
        print("--- %s seconds ---" % (time.time() - start_time))
        with open(str(pathlib.Path('../pickles/gfrozen2d.pickle').absolute()), 'wb') as f:
            pickle.dump(xzdata, f)

    fig, ax = plt.subplots(1, 1)
    levels = np.linspace(0, 2, 40)
    cs1 = ax.contourf(X, Y, xzdata, levels=levels, cmap=cm.hot)
    fig.colorbar(cs1, format="%.2f")
    ax.set_aspect('equal', 'box')

    plt.show()


def test_plot_3d_frozen():
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    electric_i_tm = SphericalField(radial=glmt.radial_electric_i_tm,
                                   theta=glmt.theta_electric_i_tm,
                                   phi=glmt.phi_electric_i_tm)
    electric_i_te = SphericalField(radial=zero, theta=glmt.theta_electric_i_te,
                                   phi=glmt.phi_electric_i_te)
    electric_i = electric_i_te + electric_i_tm

    cartesian_electric_i = CartesianField(spherical=electric_i)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    x = np.linspace(-10E-6 / 1E-6, 10E-6 / 1E-6, NUM/10)
    y = np.linspace(0, 100, NUM/10)
    X, Y = np.meshgrid(x, y)
    try:
        print('Loading results: ')
        with open(str(pathlib.Path('../pickles/gfrozen3d3.pickle').absolute()), 'rb') as f:
            Z = pickle.load(f)
    except FileNotFoundError:
        print('There are no saved results. Calculating...')
        start_time= time.time()
        Z = pow(np.vectorize(cartesian_electric_i.abs)(x=1E-6*X, y=0, z=1E-6*Y, wave_number_k=WAVE_NUMBER), 2)
        print('---- %s seconds ----' % (time.time()-start_time))
        # Saving the object:
        with open(str(pathlib.Path('../pickles/gfrozen3d3.pickle').absolute()), 'wb') as f:
            pickle.dump(Z, f)
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.inferno,
                           linewidth=0, antialiased=True)

    # Customize the z axis.
    #ax.set_zlim(-0.01, max(Z.all()))
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=15, label='|E|² [v²/m²]')
    ax.set_xlabel('x [um]')
    ax.set_ylabel('z [um]')


    plt.show()

def test_frozen_wave():
    electric_i_tm = SphericalField(radial=glmt.radial_electric_i_tm,
                                   theta=glmt.theta_electric_i_tm,
                                   phi=glmt.phi_electric_i_tm)
    electric_i_te = SphericalField(radial=zero, theta=glmt.theta_electric_i_te,
                                   phi=glmt.phi_electric_i_te)
    electric_i = electric_i_te + electric_i_tm

    cartesian_electric_i = CartesianField(spherical=electric_i)

    start_time = time.time()
    t = np.linspace(START / 1E-6, STOP / 1E-6, num=NUM)
    s = []
    for j in t:
        s.append(pow(cartesian_electric_i.abs(x=j * 1E-6, y=1E-100, z=1E-100, wave_number_k=WAVE_NUMBER), 2))
    plt.plot(t, s)
    print("::: CARTESIAN :::")
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.show()

def test_frozen_bsc():
    t = range(1, 500)
    s = []
    for i in t:
        start_time = time.time()
        glmt.beam_shape_g(i, 1, mode='TE')
        s.append(time.time() - start_time)
    plt.plot(t, s)
    plt.show()

def test_frozen_slices():
    electric_i_tm = SphericalField(radial=glmt.radial_electric_i_tm,
                                   theta=glmt.theta_electric_i_tm,
                                   phi=glmt.phi_electric_i_tm)
    electric_i_te = SphericalField(radial=zero, theta=glmt.theta_electric_i_te,
                                   phi=glmt.phi_electric_i_te)
    electric_i = electric_i_te + electric_i_tm

    cartesian_electric_i = CartesianField(spherical=electric_i)

    start_time = time.time()
    t = np.linspace(-10, 10, num=NUM)
    s40 = []
    s60 = []
    try:
        with open(str(pathlib.Path('../pickles/gcart_slice_fw.pickle').absolute()), 'rb') as f:
            print('Loading results: ')
            s40, s60 = pickle.load(f)
            plt.plot(t, s40, 'navy', label='z = 40 um')
            plt.plot(t, s60, 'firebrick', label='z = 60 um')
    except FileNotFoundError:
        print('There are no saved results. Calculating...')
        start_time = time.time()
        count = 0
        for j in t:
            s40.append(cartesian_electric_i.abs(x=j * 1E-6, y=0, z=40E-6, wave_number_k=WAVE_NUMBER))
            count += 1
            print("j = ", j, " -> ", get_max_it(np.sqrt((j * 1E-6) ** 2 + 40E-6 ** 2)))
        plt.plot(t, s40, 'navy', label='z = 40 um')

        print("::: 40 um :::")
        print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        for j in t:
            s60.append(cartesian_electric_i.abs(x=j * 1E-6, y=0, z=60E-6, wave_number_k=WAVE_NUMBER))
        plt.plot(t, s60, 'firebrick', label='z = 60 um')

        print("::: 60 um :::")
        print("--- %s seconds ---" % (time.time() - start_time))
        with open(str(pathlib.Path('../pickles/gcart_slice_fw.pickle').absolute()), 'wb') as f:
            pickle.dump((s40, s60), f)

    plt.ylabel('Magnitude do campo elétrico [V/m]')
    plt.xlabel('x [micrômetros]')
    plt.legend(loc=1)
    plt.show()


def test_cartesian_components():
    electric_i_tm = SphericalField(radial=glmt.radial_electric_i_tm,
                                   theta=glmt.theta_electric_i_tm,
                                   phi=glmt.phi_electric_i_tm)
    electric_i_te = SphericalField(radial=zero, theta=glmt.theta_electric_i_te,
                                   phi=glmt.phi_electric_i_te)
    electric_i = electric_i_te + electric_i_tm

    cartesian_electric_i = CartesianField(spherical=electric_i)

    start_time = time.time()
    t = np.linspace(0, 100, num=NUM)
    sx = []
    sy = []
    sz = []
    try:
        with open(str(pathlib.Path('../pickles/frozen_comp.pickle').absolute()), 'rb') as f:
            print('Loading results: ')
            sx, sy, sz = pickle.load(f)
            plt.plot(t, sx, 'tomato', label='ex')
            plt.plot(t, sy, 'navy', label='ey')
            plt.plot(t, sz, 'firebrick', label='ez')
    except FileNotFoundError:
        print('There are no saved results. Calculating...')

        start_time = time.time()
        for j in t:
            sx.append(pow(abs(cartesian_electric_i.functions['x'](x=0, y=0, z=j * 1E-6, wave_number_k=WAVE_NUMBER)),2))
            print("j = ", j, " -> ", get_max_it(j * 1E-6), ": ", sx[-1])
        plt.plot(t, sx, 'tomato', label='hx')
        print("::: X :::")
        print("--- %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        for j in t:
            sy.append(pow(abs(cartesian_electric_i.functions['y'](x=0, y=0, z=j * 1E-6, wave_number_k=WAVE_NUMBER)),2))
            print("j = ", j, " -> ", get_max_it(j * 1E-6), ": ", sy[-1])
        plt.plot(t, sy, 'navy', label='hy')
        print("::: Y :::")
        print("--- %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        for j in t:
            sz.append(pow(abs(cartesian_electric_i.functions['z'](x=0, y=0, z=j * 1E-6, wave_number_k=WAVE_NUMBER)),2))
            print("j = ", j, " -> ", get_max_it(j * 1E-6), ": ", sz[-1])
        plt.plot(t, sz, 'firebrick', label='hz')
        print("::: Z :::")
        print("--- %s seconds ---" % (time.time() - start_time))
        with open(str(pathlib.Path('../pickles/frozen_comp.pickle').absolute()), 'wb') as f:
            pickle.dump((sx, sy, sz), f)

    plt.ylabel('Magnitude do campo elétrico [V/m]')
    plt.xlabel('x [micrômetros]')
    plt.legend(loc=1)
    plt.show()

MAX_IT = get_max_it(STOP)


#test_h_field_vs_bessel()
#test_e_field_vs_bessel()
test_radial_convergence()
#test_cartesian_fields()
#test_frozen_wave()
#test_plot_2d_bessel()
#test_plot_3d_frozen()
#test_plot_2d()
#test_cartesian_components()
#test_cartesian_fields()
#test_fields()
#test_frozen_bsc()
#test_frozen_slices()

