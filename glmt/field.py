# -*- coding: utf-8 -*-
"""
Module for representing fields in spherical and cartesian coordinates.

@author: Luiz Felipe Machado Votto
"""

from abc import ABC, abstractmethod
import numpy as np

def function_sum(function_1, function_2):
    """ Returns the addition of 2 functions """
    def summed_function(*args, **kwargs):
        return function_1(*args, **kwargs) + function_2(*args, **kwargs)

    return summed_function

class Field(ABC):
    """ This is representative of a Field in tridimensional space.
    """
    functions = {}

    def __init__(self, **kwargs):
        self.functions = kwargs

    @abstractmethod
    def __add__(self, other):
        pass

    def evaluate(self, *args, **kwargs):
        """ Evaluates the value of the field given a point.
        """
        result = []
        for key in kwargs:
            try:
                result.append(self.functions[key](*args, **kwargs))
            except KeyError:
                continue
        return np.array(result)

    def abs(self, *args, **kwargs):
        """ Computes absolute value of field """
        return np.linalg.norm(abs(self.evaluate(*args, **kwargs)))


class SphericalField(Field):
    """ Represents a tridimensional field in spherical coordinates
    """
    def __init__(self, radial=None, theta=None, phi=None):
        kwargs = {'radial' : radial, 'theta' : theta, 'phi' : phi}
        super(SphericalField, self).__init__(**kwargs)

    def __add__(self, other):
        if isinstance(other, Field):
            result = SphericalField()
        else:
            raise ValueError('It is impossible to add a Field'
                             ' to a non-Field object')
        for key in result.functions:
            result.functions[key] = function_sum(self.functions[key],
                                                 other.functions[key])
        return result


def spherical_in_cartesian(spherical_function):
    """ Write a function in spherical coordinates in terms
        of cartesian coordinates
    """
    def cartesian_function(x, y, z, *args, **kwargs):
        return spherical_function(cartesian_radial(x, y, z),
                                  cartesian_theta(x, y, z),
                                  cartesian_phi(x, y),
                                  *args, **kwargs)
    return cartesian_function

def cartesian_radial(x, y, z):
    return np.sqrt(x * x + y * y + z * z)

def cartesian_theta(x, y, z):
    rho = np.sqrt(x * x + y * y)

    return np.arctan2(rho, z)

def cartesian_phi(x, y):
    return np.arctan2(y, x)


def spherical_to_cartesian(function_r,
                           function_theta,
                           function_phi):
    """ Convert a spherical field to cartesian coordinates """
    func_r = spherical_in_cartesian(function_r)
    func_theta = spherical_in_cartesian(function_theta)
    func_phi = spherical_in_cartesian(function_phi)

    def function_x(x, y, z, *args, **kwargs):
        return func_r(x, y, z, *args, **kwargs) \
               * np.sin(cartesian_theta(x, y, z)) \
               * np.cos(cartesian_phi(x, y)) \
               + func_theta(x, y, z, *args, **kwargs) \
               * np.cos(cartesian_theta(x, y, z)) \
               * np.cos(cartesian_phi(x, y)) \
               + func_phi(x, y, z, *args, **kwargs) \
               * (-np.sin(cartesian_theta(x, y, z)))

    def function_y(x, y, z, *args, **kwargs):
        return func_r(x, y, z, *args, **kwargs) \
               * np.sin(cartesian_theta(x, y, z)) \
               * np.sin(cartesian_phi(x, y)) \
               + func_theta(x, y, z, *args, **kwargs) \
               * np.cos(cartesian_theta(x, y, z)) \
               * np.sin(cartesian_phi(x, y)) \
               + func_phi(x, y, z, *args, **kwargs) \
               * np.cos(cartesian_theta(x, y, z))

    def function_z(x, y, z, *args, **kwargs):
        return func_r(x, y, z, *args, **kwargs) \
               * np.cos(cartesian_theta(x, y, z)) \
               + func_theta(x, y, z, *args, **kwargs) \
               * (-np.sin(cartesian_theta(x, y, z))) \

    return function_x, function_y, function_z


class CartesianField(Field):
    """ Represents a tridimensional field in cartesian coordinates
    """
    def __init__(self, x=None, y=None, z=None, spherical=None):
        if spherical:
            x, y, z = spherical_to_cartesian(spherical.functions['radial'],
                                             spherical.functions['theta'],
                                             spherical.functions['phi'])
        kwargs = {'x' : x, 'y' : y, 'z' : z}
        super(CartesianField, self).__init__(**kwargs)

    def __add__(self, other):
        if isinstance(other, SphericalField):
            other = CartesianField(spherical=other)
        if isinstance(other, Field):
            result = CartesianField()
        else:
            raise ValueError('It is impossible to add a Field'
                             ' to a non-Field object')
        for key in result.functions:
            result.functions[key] = function_sum(self.functions[key],
                                                 other.functions[key])
        return result
