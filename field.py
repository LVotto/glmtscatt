# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 15:35:56 2017

@author: luiz_
"""

from abc import ABC
import numpy as np

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

    def abs(self, x1_value, x2_value, x3_value):
        return np.linalg.norm(
            self.evaluate(x=x1_value, y=x2_value, z=x3_value))

def spherical_in_cartesian(spherical_function):
    def cartesian_function(x_value, y_value, z_value, *args, **kwargs):
        return spherical_function(np.sqrt(pow(x_value, 2) \
                                  + pow(y_value, 2) \
                                  + pow(z_value, 2)),
                          np.arctan(np.sqrt(pow(x_value, 2) \
                                            + pow(y_value, 2)) \
                                    / z_value),
                          np.arctan(y_value / z_value),
                          *args, **kwargs)
    return cartesian_function

def spherical_to_cartesian(function_r,
                           function_theta,
                           function_phi):
    def function_x(x_value, y_value, z_value, *args, **kwargs):
        r = np.sqrt(pow(x_value, 2), pow(y_value, 2), pow(z_value, 2))
        rho = np.sqrt(pow(x_value, 2), pow(y_value, 2))
        
        func_r = spherical_in_cartesian(function_r)(x_value,
                                                    y_value,
                                                    z_value,
                                                    *args,
                                                    **kwargs)
        func_theta = spherical_in_cartesian(function_theta)(x_value,
                                                            y_value,
                                                            z_value,
                                                            *args,
                                                            **kwargs)
        func_phi = spherical_in_cartesian(function_phi)(x_value,
                                                        y_value,
                                                        z_value,
                                                        *args,
                                                        **kwargs)
    
        return x_value / r * func_r \
               + x_value * z_value / (r * rho) * func_theta \
               - y_value / rho * func_phi
               
    def function_y(x_value, y_value, z_value, *args, **kwargs):
        r = np.sqrt(pow(x_value, 2), pow(y_value, 2), pow(z_value, 2))
        rho = np.sqrt(pow(x_value, 2), pow(y_value, 2))
        
        func_r = spherical_in_cartesian(function_r)(x_value,
                                                    y_value,
                                                    z_value,
                                                    *args,
                                                    **kwargs)
        func_theta = spherical_in_cartesian(function_theta)(x_value,
                                                            y_value,
                                                            z_value,
                                                            *args,
                                                            **kwargs)
        func_phi = spherical_in_cartesian(function_phi)(x_value,
                                                        y_value,
                                                        z_value,
                                                        *args,
                                                        **kwargs)
        
        return y_value / r * func_r \
               + y_value * z_value / (r * rho) * func_theta \
               + x_value / rho * func_phi
               
    def function_z(x_value, y_value, z_value, *args, **kwargs):
        r = np.sqrt(pow(x_value, 2), pow(y_value, 2), pow(z_value, 2))
        rho = np.sqrt(pow(x_value, 2), pow(y_value, 2))
        
        func_r = spherical_in_cartesian(function_r)(x_value,
                                                    y_value,
                                                    z_value,
                                                    *args,
                                                    **kwargs)
        func_theta = spherical_in_cartesian(function_theta)(x_value,
                                                            y_value,
                                                            z_value,
                                                            *args,
                                                            **kwargs)
    
        return z_value / r * func_r \
               + rho / r * func_theta
    
    return function_x, function_y, function_z
    
class SphericalField(Field):
    """ Represents a tridimensional field in spherical coordinates
    """
    def __init__(self, r=None, theta=None, phi=None):
        kwargs = {'r' : r, 'theta' : theta, 'phi' : phi}
        super(SphericalField, self).__init__(**kwargs)


class CartesianField(Field):
    """ Represents a tridimensional field in cartesian coordinates
    """
    def __init__(self, x=None, y=None, z=None, spherical=None):
        if isinstance(spherical, SphericalField):
            x, y, z = spherical_to_cartesian(spherical.functions['r'],
                                             spherical.functions['theta'],
                                             spherical.functions['phi'])
        kwargs = {'x' : x, 'y' : y, 'z' : z}
        super(CartesianField, self).__init__(**kwargs)