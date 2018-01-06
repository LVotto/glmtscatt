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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import glmt.glmt as glmt
from glmt.field import SphericalField, CartesianField
from glmt.specials import squared_bessel_0
from glmt.utils import zero, get_max_it, normalize_list, open_file, PlotHandler
from glmt.constants import AXICON, WAVE_NUMBER

#STOP = 10/(WAVE_NUMBER * np.sin(AXICON))  # for bessel beam
STOP = 100E-6
#STOP = 10/(WAVE_NUMBER) # (to see the peak)
#START = 0
START = -STOP
NUM = 500

def plot_square_abs_in_z(x, start=START, stop=STOP, num=NUM, pickle_file='cache'):
    e = declare_cartesian_electric_field()
    t = np.linspace(start, stop, num)
    
    try:
        with open(str(pathlib.Path('../pickles/%s.pickle' % pickle_file).absolute()), 'rb') as f:
            print('Loading results: ')
            sz = pickle.load(f)
            print(sz)
            for item in sz:
                if np.isnan(item):
                    item = 0
                    print(sz)
            plt.plot(len(sz), sz, 'firebrick')
    except FileNotFoundError:
        print('There are no saved results. Calculating...')
        sz = []
        for j in t:
            sz.append(pow(e.abs(x=x, y=0, z=j * 1E-6,
                               wave_number_k=WAVE_NUMBER),
                         2)
                    )
            print("j = ", j, " -> ", get_max_it(abs(j) * 1E-6), ": ", sz[-1])
    plt.plot(t, sz)
    plt.xlabel('z [micrômetros]')
    plt.ylabel('|E|² [V²/m²]')
    plt.show()
    with open(str(pathlib.Path('../pickles/%s.pickle' % pickle_file).absolute()), 'wb') as f:
        pickle.dump(sz, f)
    
    
def plot_square_abs_in_x(start=START, stop=STOP, num=NUM):
    e = declare_cartesian_electric_field()
    
    t = np.linspace(start, stop, num)
    s = []
    for j in t:
        s.append(pow(e.abs(x=j * 1E-6, y=0, z=0,
                           wave_number_k=WAVE_NUMBER),
                     2)
                )
    plt.plot(t, s)
    plt.show()
    return t, s

def declare_spherical_electric_field():
    electric_i_tm = SphericalField(radial=glmt.radial_electric_i_tm,
                                   theta=glmt.theta_electric_i_tm,
                                   phi=glmt.phi_electric_i_tm)
    electric_i_te = SphericalField(radial=zero, theta=glmt.theta_electric_i_te,
                                   phi=glmt.phi_electric_i_te)
    return electric_i_te + electric_i_tm

def declare_cartesian_electric_field():
    return CartesianField(spherical=declare_spherical_electric_field())

def declare_cartesian_magnetic_field():
    magnetic_i_tm = SphericalField(radial=zero,
                                   theta=glmt.theta_magnetic_i_tm,
                                   phi=glmt.phi_magnetic_i_tm)
    magnetic_i_te = SphericalField(radial=glmt.radial_magnetic_i_te,
                                   theta=glmt.theta_magnetic_i_te,
                                   phi=glmt.phi_magnetic_i_te)
    magnetic_i = magnetic_i_te + magnetic_i_tm

    return CartesianField(spherical=magnetic_i)


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
    end = 1000
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
        plt.title(x / 1E-6)
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
    tx = np.linspace(-10, 10, num=NUM/4)
    ty = tx
    tz = np.linspace(0, 450, num=NUM/4)
    sx = []
    sy = []
    sz = []
    try:
        with open(str(pathlib.Path('../pickles/gfw_cart_s.pickle').absolute()), 'rb') as f:
            print('Loading results: ')
            sx, sy, sz = pickle.load(f)
            plt.plot(tx, sx, 'tomato', label='eixo-x')
            plt.plot(ty, sy, 'navy', label='eixo-y', linestyle='--')
            #plt.plot(tz, sz, 'firebrick', label='eixo-z')
    except FileNotFoundError:
        print('There are no saved results. Calculating...')
        start_time = time.time()
        for j in tx:
            if j == 0:
                j = 1E-16
            sx.append(pow(abs(cartesian_electric_i.abs(x=j * 1E-6, y=0, z=0, wave_number_k=WAVE_NUMBER)),2))
            print("j = ", j, " -> ", get_max_it(abs(j) * 1E-6), ": ", sx[-1])
        plt.plot(tx, sx, 'tomato', label='eixo-x')
        print("::: X :::")
        print("--- %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        for j in ty:
            if j == 0:
                j = 1E-16
            sy.append(pow(abs(cartesian_electric_i.abs(x=0, y=j * 1E-6, z=0, wave_number_k=WAVE_NUMBER)),2))
            print("j = ", j, " -> ", get_max_it(abs(j) * 1E-6), ": ", sy[-1])
        plt.plot(ty, sy, 'navy', label='eixo-y')
        print("::: Y :::")
        print("--- %s seconds ---" % (time.time() - start_time))

        print(sz)
        start_time = time.time()
        for j in tz:
            if j == 0:
                j = 1E-16
            sz.append(pow(abs(cartesian_electric_i.abs(x=0, y=0, z=j * 1E-6, wave_number_k=WAVE_NUMBER)),2))
            print("j = ", j, " -> ", get_max_it(abs(j) * 1E-6), ": ", sz[-1])
        plt.plot(tz, sz, 'firebrick', label='eixo-z')
        print(sz)
        print("::: Z :::")
        print("--- %s seconds ---" % (time.time() - start_time))
        with open(str(pathlib.Path('../pickles/gfw_cart_s.pickle').absolute()), 'wb') as f:
            pickle.dump((sx, sy, sz), f)

    plt.ylabel('|E|² [V²/m²]')
    plt.xlabel('r [micrômetros]')
    plt.legend(loc=1)
    plt.show()

def test_cartesian_z():
    electric_i_tm = SphericalField(radial=glmt.radial_electric_i_tm,
                                   theta=glmt.theta_electric_i_tm,
                                   phi=glmt.phi_electric_i_tm)
    electric_i_te = SphericalField(radial=zero, theta=glmt.theta_electric_i_te,
                                   phi=glmt.phi_electric_i_te)
    electric_i = electric_i_te + electric_i_tm

    cartesian_electric_i = CartesianField(spherical=electric_i)

    start_time = time.time()
    t = np.linspace(0, 100, num=NUM/4)
    sz = []
    try:
        with open(str(pathlib.Path('../pickles/frozen_cartesian5.pickle').absolute()), 'rb') as f:
            print('Loading results: ')
            sz = pickle.load(f)
            print(sz)
            for item in sz:
                if np.isnan(item):
                    item = 0
                    print(sz)
            plt.plot(t, sz, 'firebrick', label='eixo-z')
    except FileNotFoundError:
        print('There are no saved results. Calculating...')
        print(sz)
        start_time = time.time()
        for j in t:
            if j == 0:
                j = 1E-16
            sz.append(pow(abs(cartesian_electric_i.abs(x=0, y=0, z=j * 1E-6, wave_number_k=WAVE_NUMBER)),2))
            print("j = ", j, " -> ", get_max_it(abs(j) * 1E-6), ": ", sz[-1])
        plt.plot(t, sz, 'firebrick', label='eixo-z')
        print(sz)
        print("::: Z :::")
        print("--- %s seconds ---" % (time.time() - start_time))
        with open(str(pathlib.Path('../pickles/frozen_cartesian5.pickle').absolute()), 'wb') as f:
            pickle.dump(sz, f)

    plt.ylabel('Magnitude do campo elétrico [V/m]')
    plt.xlabel('z [micrômetros]')
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

    x = np.linspace(-10, 10, 300)
    y = np.linspace(0, 450, 300)
    X, Y = np.meshgrid(x, y)

    start_time = time.time()
    try:
        print('Searching for results')
        with open(str(pathlib.Path('../pickles/ggfw3300a.pickle').absolute()), 'rb') as f:
            xzdata = pickle.load(f)
            print('Results where found!')
    except FileNotFoundError:
        print('Results not found. Calculating...')
        xzdata = abs(np.vectorize(cartesian_electric_i.abs)(x=1E-6*X, y=0, z=1E-6*Y, wave_number_k=WAVE_NUMBER))
        print("--- %s seconds ---" % (time.time() - start_time))
        with open(str(pathlib.Path('../pickles/frozenlast2.pickle').absolute()), 'wb') as f:
            pickle.dump(xzdata, f)

    fig, ax = plt.subplots(1, 1)
    levels = np.linspace(0, 2, 40)
    cs1 = ax.contourf(X, Y, xzdata, levels=levels, cmap=cm.hot)
    with open(str(pathlib.Path('../pickles/ggfw3300a.pickle').absolute()), 'wb') as f:
        pickle.dump(xzdata, f)

    fig, ax = plt.subplots(1, 1)
    #levels = np.linspace(0, 8, 40)
    cs1 = ax.contourf(X, Y, pow(xzdata, 2), cmap=cm.inferno)
    #ax.set_aspect('equal', 'box')
    fig.colorbar(cs1, format="%.2f")
    ax.set_aspect('equal', 'box')
    fig.colorbar(cs1, format="%.2f")
    ax.set_xlabel('x [micrômetros]')
    ax.set_ylabel('z [micrômetros]')

    plt.show()


def test_plot_3d_frozen():
    electric_i_tm = SphericalField(radial=glmt.radial_electric_i_tm,
                                   theta=glmt.theta_electric_i_tm,
                                   phi=glmt.phi_electric_i_tm)
    electric_i_te = SphericalField(radial=zero, theta=glmt.theta_electric_i_te,
                                   phi=glmt.phi_electric_i_te)
    electric_i = electric_i_te + electric_i_tm

    cartesian_electric_i = CartesianField(spherical=electric_i)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    print("I am plotting the Bessel Beam's z component in 3D...")

    # Make data.
    x = np.linspace(-200, 200, 125)
    y = np.linspace(-200, 200, 125)
    X, Y = np.meshgrid(x, y)
    try:
        print('Loading results: ')
        with open(str(pathlib.Path('../pickles/besselz3d.pickle').absolute()), 'rb') as f:
            Z = pickle.load(f)
            print(len(Z))
            x = np.linspace(-200, 200, len(Z))
            y = np.linspace(-200, 200, len(Z))
            X, Y = np.meshgrid(x, y)
    except FileNotFoundError:
        print('There are no saved results. Calculating...')
        start_time= time.time()
        Z = pow(abs(np.vectorize(cartesian_electric_i.functions['z'])(x=1E-6*X, y=1E-6*Y, z=0, wave_number_k=WAVE_NUMBER)), 2)
        print('---- %s seconds ----' % (time.time()-start_time))
        # Saving the object:

        with open(str(pathlib.Path('../pickles/frozen3dultimate.pickle').absolute()), 'wb') as f:
            pickle.dump(Z, f)
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis,
                           linewidth=1, antialiased=True)

    # Customize the z axis.
    #ax.set_zlim(0, 8)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

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

def test_frozen_bsc(mode='TM'):
    t = range(1, 500)
    s = []
    for i in t:
        g = glmt.beam_shape_g(i, 1, mode=mode)
        s.append(abs(g))
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
    t = np.linspace(-10, 10, num=NUM/5)
    s40 = []
    s60 = []
    try:
        with open(str(pathlib.Path('../pickles/gcart_slice500.pickle').absolute()), 'rb') as f:
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
            print("j = ", j, " -> ", get_max_it(np.sqrt((j * 1E-6) ** 2 + 40E-6 ** 2)), " : ", s40[-1])
        plt.plot(t, s40, 'navy', label='z = 40 um')

        print("::: 40 um :::")
        print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        for j in t:
            s60.append(cartesian_electric_i.abs(x=j * 1E-6, y=0, z=60E-6, wave_number_k=WAVE_NUMBER))
            print("j = ", j, " -> ", get_max_it(np.sqrt((j * 1E-6) ** 2 + 60E-6 ** 2)), " : ", s60[-1])
        plt.plot(t, s60, 'firebrick', label='z = 60 um')

        print("::: 60 um :::")
        print("--- %s seconds ---" % (time.time() - start_time))
        with open(str(pathlib.Path('../pickles/gcart_slice500.pickle').absolute()), 'wb') as f:
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
    t = np.linspace(0, 100, num=NUM/4)
    sx = []
    sy = []
    sz = []
    try:
        with open(str(pathlib.Path('../pickles/bessel_comp.pickle').absolute()), 'rb') as f:

            print('Loading results: ')
            sx, sy, sz = pickle.load(f)
            plt.plot(t, sx, 'tomato', label='Ex')
            plt.plot(t, sy, 'navy', label='Ey', linestyle='--')
            plt.plot(t, sz, 'firebrick', label='Ez')
    except FileNotFoundError:
        print('There are no saved results. Calculating...')
        start_time = time.time()
        for j in t:
            if j == 0:
                j = 1E-16
            sx.append(pow(abs(cartesian_electric_i.functions['x'](x=0, y=0, z=j * 1E-6, wave_number_k=WAVE_NUMBER)),2))
            print("j = ", j, " -> ", get_max_it(abs(j) * 1E-6), ": ", sx[-1])
        plt.plot(t, sx, 'tomato', label='Ex')
        print("::: X :::")
        print("--- %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        for j in t:
            if j == 0:
                j = 1E-16
            sy.append(pow(abs(cartesian_electric_i.functions['y'](x=0, y=0, z=j * 1E-6, wave_number_k=WAVE_NUMBER)),2))
            print("j = ", j, " -> ", get_max_it(abs(j) * 1E-6), ": ", sy[-1])
        plt.plot(t, sy, 'navy', label='Ey')
        print("::: Y :::")
        print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        for j in t:
            if j == 0:
                j = 1E-16
            sz.append(pow(abs(cartesian_electric_i.functions['z'](x=0, y=0, z=j * 1E-6, wave_number_k=WAVE_NUMBER)),2))
            print("j = ", j, " -> ", get_max_it(abs(j) * 1E-6), ": ", sz[-1])
        plt.plot(t, sz, 'firebrick', label='Ez')
        print("::: Z :::")
        print("--- %s seconds ---" % (time.time() - start_time))
        with open(str(pathlib.Path('../pickles/frozen_compZZZ.pickle').absolute()), 'wb') as f:
            pickle.dump((sx, sy, sz), f)

    plt.ylabel('|E|² [V²/m²]')
    plt.xlabel('x [micrômetros]')
    plt.legend(loc=1)
    plt.show()

def test_pickles():
    import random
    try:
        with open(str(pathlib.Path('../pickles/test.pickle').absolute()), 'rb') as f:
            a = pickle.load(f)
            print('LOADED: ', a)
    except FileNotFoundError:
        with open(str(pathlib.Path('../pickles/test.pickle').absolute()), 'wb') as f:
            a = random.choice(range(0, 10000))
            print('CREATED: ', a)
            pickle.dump(a, f)

def test_some_axes():
    f = declare_cartesian_electric_field()

    assert abs(f.functions['x'](x=0, y=1E-6, z=0, wave_number_k=WAVE_NUMBER) \
               - f.functions['x'](x=1E-6, y=0, z=0, wave_number_k=WAVE_NUMBER)) < 1E-6
    assert abs(f.functions['x'](x=0, y=1E-6, z=0, wave_number_k=WAVE_NUMBER) \
               - f.functions['x'](x=np.sqrt(2)/2 * 1E-6, y=np.sqrt(2)/2 * 1E-6, z=0, wave_number_k=WAVE_NUMBER)) < 1E-6


def test_magnetic_components():
    cartesian_magnetic_i = declare_cartesian_magnetic_field()

    start_time = time.time()
    t = np.linspace(0, 100, num=NUM)
    sx = []
    sy = []
    sz = []
    try:
        with open(str(pathlib.Path('../pickles/bessel_mag_comp.pickle').absolute()), 'rb') as f:
            print('Loading results: ')
            sx, sy, sz = pickle.load(f)
            #plt.plot(t, sx, 'tomato', label='hx')
            #plt.plot(t, sy, 'navy', label='hy')
            plt.plot(t, sz, 'firebrick', label='hz')
    except FileNotFoundError:
        print('There are no saved results. Calculating...')

        start_time = time.time()
        for j in t:
            sx.append(pow(abs(cartesian_magnetic_i.functions['x'](x=0, y=j * 1E-6, z=0, wave_number_k=WAVE_NUMBER)),2))
            print("j = ", j, " -> ", get_max_it(abs(j) * 1E-6), ": ", sx[-1])
        plt.plot(t, sx, 'tomato', label='hx')
        print("::: X :::")
        print("--- %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        for j in t:
            sy.append(pow(abs(cartesian_magnetic_i.functions['y'](x=0, y=j * 1E-6, z=0, wave_number_k=WAVE_NUMBER)),2))
            print("j = ", j, " -> ", get_max_it(abs(j) * 1E-6), ": ", sy[-1])
        plt.plot(t, sy, 'navy', label='hy')
        print("::: Y :::")
        print("--- %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        for j in t:
            sz.append(pow(abs(cartesian_magnetic_i.functions['z'](x=0, y=j * 1E-6, z=0, wave_number_k=WAVE_NUMBER)),2))
            print("j = ", j, " -> ", get_max_it(abs(j) * 1E-6), ": ", sz[-1])
        plt.plot(t, sz, 'firebrick', label='hz')
        print("::: Z :::")
        print("--- %s seconds ---" % (time.time() - start_time))
        with open(str(pathlib.Path('../pickles/bessel_mag_comp.pickle').absolute()), 'wb') as f:
            pickle.dump((sx, sy, sz), f)

    plt.ylabel('|H|² [V²/m²]')
    plt.xlabel('x [micrômetros]')
    plt.legend(loc=1)
    plt.show()

def test_circle_bessel():
    f = declare_spherical_electric_field()

    t = np.linspace(0, 2*np.pi, 100)
    s = []

    for j in t:
        s.append(abs(f.functions['phi'](radial=1E-6, theta=np.pi/2, phi=j, wave_number_k=WAVE_NUMBER))**2 \
                     +abs(f.functions['radial'](radial=1E-6, theta=np.pi/2, phi=j, wave_number_k=WAVE_NUMBER))**2)
        print('j = ', j, " -> ", s[-1])
    plt.plot(t, s)
    plt.show


def test_increment_decay():
    from glmt.specials import _riccati_bessel_j, d2_riccati_bessel_j, legendre_p
    from glmt.glmt import plane_wave_coefficient, beam_shape_g
    max_it = 1000
    r = np.linspace(0, 100E-6, 5)
    wave_number_k = WAVE_NUMBER
    theta = np.pi/2
    phi = 0
    
    for radial in r:
        riccati_bessel_list = _riccati_bessel_j(max_it,
                                                wave_number_k * radial)
        riccati_bessel = riccati_bessel_list[0]
        s = []
        result = 0
        n = 1
        print(radial)
        while n < max_it:
            for m in [-1, 1]:
                increment = plane_wave_coefficient(n, wave_number_k) \
                          * beam_shape_g(n, m, mode='TM') \
                          * (d2_riccati_bessel_j(n, wave_number_k * radial) \
                             + riccati_bessel[n]) \
                          * legendre_p(n, abs(m), np.cos(theta)) \
                          * np.exp(1j * m * phi)
                result += increment
            s.append(abs(wave_number_k * result))
            n += 1
        plt.plot(range(1, max_it), s)
        plt.title("%s micrômetros" % (radial / 1E-6))
        plt.xlabel('N')
        plt.ylabel('|E| [V/m]')
        N = get_max_it(radial)
        plt.plot(N, s[N], 'rx', label='N(r) = %s' % N)
        plt.legend()
        print(result)
        plt.show()

def plot_n_max(max_radial=1000, num=500):
    t = np.linspace(0, max_radial, num=num)
    s = []
    for j in t:
        s.append(get_max_it(j * 1E-6))
    plt.plot(t, s)
    plt.xlabel('r [micrômetros]')
    plt.ylabel('N(r)')
    plt.show()
    
def plot_3d_xz(min_z=-22E-6,max_z=22E-6, min_x=-10E-6, max_x=10E-6, num=400,
               load=True, file_name='2d_zx', cmap=cm.inferno):
    f = declare_cartesian_electric_field()
    
    tz = np.linspace(min_z, max_z, num)
    tx = np.linspace(min_x, max_x, num)
    X, Z = np.meshgrid(tx, tz)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    if load:
        print('::::: Loading 2D XZ graph :::::')
        with open_file(file_name=file_name) as f:
            F = pickle.load(f)
    else:
        start_time = time.time()
        print('::::: Plotting 2D XZ graph :::::')
        F = np.vectorize(f.abs)(x=X, y=0, z=Z, wave_number_k=WAVE_NUMBER)
        print("--- %s seconds ---" % (time.time() - start_time))
        with open_file(file_name=file_name, operation='wb') as f:
            pickle.dump(F, f)
    
    # Plot the surface.
    surf = ax.plot_surface(X, Z, F * F, cmap=cmap, antialiased=True)
    # Customize the z axis.
    #ax.set_zlim(0, 8)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=15, label='|E|² [v²/m²]')
    ax.set_xlabel('x [um]')
    ax.set_ylabel('z [um]')
    plt.show()    
    
    contourf = plt.contourf(X, Z, F * F, cmap=cmap)
    fig.colorbar(contourf, shrink=0.5, aspect=15, label='|E|² [v²/m²]')
    plt.show()    
    
    return X, Z, F

def plot_component_in_z(x, start=START, stop=STOP, num=NUM,
                        load=False, pickle_file='cache', component='x'):
    e = declare_cartesian_electric_field()
    t = np.linspace(start, stop, num)
    
    if load:
        with open(str(pathlib.Path('../pickles/%s.pickle' % pickle_file).absolute()), 'rb') as f:
            print('Loading results: ')
            sz = pickle.load(f)
            print(sz)
            for item in sz:
                if np.isnan(item):
                    item = 0
                    print(sz)
            plt.plot(len(sz), sz, 'firebrick')
    else:
        print('Calculating...')
        sz = []
        for j in t:
            sz.append(pow(e[component].abs(x=x, y=0, z=j * 1E-6,
                               wave_number_k=WAVE_NUMBER),
                         2)
                    )
            print("j = ", j, " -> ", get_max_it(abs(j) * 1E-6), ": ", sz[-1])
    plt.plot(t, sz)
    plt.xlabel('z [micrômetros]')
    plt.ylabel('|E|² [V²/m²]')
    plt.show()
    with open(str(pathlib.Path('../pickles/%s.pickle' % pickle_file).absolute()), 'wb') as f:
        pickle.dump(sz, f)
        
def square_abs_z(x, field=declare_cartesian_electric_field(), 
                 start=START, stop=STOP, num=NUM, load=False,
                 pickle_file='cache', component='x'):
    e = field
    t = np.linspace(start, stop, num)
    
    if load:
        with open(str(pathlib.Path('../pickles/%s.pickle' % pickle_file).absolute()), 'rb') as f:
            print('Loading results: ')
            sz = pickle.load(f)
            print(sz)
            for item in sz:
                if np.isnan(item):
                    item = 0
                    print(sz)
            plt.plot(len(sz), sz, 'firebrick')
    else:
        print('Calculating...')
        sz = []
        for j in t:
            sz.append(pow(e[component].abs(x=x, y=0, z=j * 1E-6,
                               wave_number_k=WAVE_NUMBER),
                         2)
                    )
            print("j = ", j, " -> ", get_max_it(abs(j) * 1E-6), ": ", sz[-1])
    plt.plot(t, sz)
    plt.xlabel('z [micrômetros]')
    plt.ylabel('|E|² [V²/m²]')
    plt.show()
    with open(str(pathlib.Path('../pickles/%s.pickle' % pickle_file).absolute()), 'wb') as f:
        pickle.dump(sz, f)
        
def plot_and_store_json():
    t, s = plot_square_abs_in_x(start=0, stop=100, num=400)
    plot_handler = PlotHandler(path='grafbessel.json',
                               data=[t, s],
                               title='Módulo ao quadrado do feixe de Bessel',
                               labels=['x [micrômetros]', 
                                       '|E|² [V²/m²]'],
                               shape=1
                              )
    plot_handler.store()
    return plot_handler.plot()


MAX_IT = get_max_it(STOP)

#test_pickles()
#test_h_field_vs_bessel()
#test_e_field_vs_bessel()
#test_radial_convergence()
#test_cartesian_fields()
#test_cartesian_z()
#test_frozen_wave()
#test_plot_2d_bessel()
#test_plot_3d_frozen()
#test_plot_2d()
#test_cartesian_components()
#test_cartesian_fields()
#test_fields()
#test_frozen_bsc()
#test_frozen_slices()
#test_some_axes()
#test_magnetic_components()
#test_circle_bessel()
#test_increment_decay()
