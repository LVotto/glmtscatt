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
        """
        if isinstance(self, SphericalField) \
            and isinstance(other, CartesianField):
            cart = spherical_to_cartesian(self)
            return function_sum(cart, other)
        elif isinstance(self, CartesianField) \
             and isinstance(other, SphericalField):
            cart = spherical_to_cartesian(other)
            return function_sum(self, cart)
        else:
            return function_sum(self, other)
        """

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
        return np.linalg.norm(self.evaluate(*args, **kwargs))

class SphericalField(Field):
    """ Represents a tridimensional field in spherical coordinates
    """
    def __init__(self, radial=None, theta=None, phi=None):
        kwargs = {'radial' : radial, 'theta' : theta, 'phi' : phi}
        super(SphericalField, self).__init__(**kwargs)
    
    def __add__(self, other):
        if isinstance(other, CartesianField):
            result = CartesianField(spherical=self)
        elif isinstance(other, Field):
            result = self
        else:
            raise ValueError('It is impossible to add a Field'
                             'with a non-Field object')
        for key in result.functions:
            result.functions[key] = function_sum(result.functions[key], 
                                                 other.functions[key])
        return result

def spherical_in_cartesian(spherical_function):
    """ Write a fucntion in spherical coordinates in terms
        of cartesian coordinates
    """
    def cartesian_function(x, y, z, *args, **kwargs):
        return spherical_function(np.sqrt(pow(x, 2) \
                                  + pow(y, 2) \
                                  + pow(z, 2)),
                                  np.arctan(np.sqrt(pow(x, 2) \
                                                    + pow(y, 2)) \
                                  / z),
                                  np.arctan(y / x),
                                  *args, **kwargs)
    return cartesian_function

def spherical_to_cartesian(function_r,
                           function_theta,
                           function_phi):
    """ Convert a spherical field to cartesian coordinates """
    func_r = spherical_in_cartesian(function_r)
    func_theta = spherical_in_cartesian(function_theta)
    func_phi = spherical_in_cartesian(function_phi)

    def function_x(x, y, z, *args, **kwargs):
        r = np.sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2))
        rho = np.sqrt(pow(x, 2) + pow(y, 2))

        return x / r * func_r(x, y, z,
                                    *args, **kwargs) \
               + x * z / (r * rho) * func_theta(x, y,
                                                            z, *args,
                                                            **kwargs) \
               - y / rho * func_phi(x, y, z,
                                          *args, **kwargs)

    def function_y(x, y, z, *args, **kwargs):
        r = np.sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2))
        rho = np.sqrt(pow(x, 2) + pow(y, 2))

        return y / r * func_r(x, y, z,
                                    *args, **kwargs) \
               + y * z / (r * rho) * func_theta(x, y,
                                                            z, *args,
                                                            **kwargs) \
               + x / rho * func_phi(x, y, z,
                                          *args, **kwargs)

    def function_z(x, y, z, *args, **kwargs):
        r = np.sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2))
        rho = np.sqrt(pow(x, 2) + pow(y, 2))

        return z / r * func_r(x, y, z,
                                    *args, **kwargs) \
               + rho / r * func_theta(x, y, z,
                                      *args, **kwargs)

    return function_x, function_y, function_z


class CartesianField(Field):
    """ Represents a tridimensional field in cartesian coordinates
    """
    def __init__(self, x=None, y=None, z=None, spherical=None):
        if isinstance(spherical, SphericalField):
            x, y, z = spherical_to_cartesian(spherical.functions['radial'],
                                             spherical.functions['theta'],
                                             spherical.functions['phi'])
        kwargs = {'x' : x, 'y' : y, 'z' : z}
        super(CartesianField, self).__init__(**kwargs)
        
    def __add__(self, other):
        if isinstance(other, SphericalField):
            result = CartesianField(spherical=other)
        elif isinstance(other, Field):
            result = self
        else:
            raise ValueError('It is impossible to add a Field'
                             'with a non-Field object')
        for key in result.functions:
            result.functions[key] = function_sum(result.functions[key], 
                                                 other.functions[key])
        return result
