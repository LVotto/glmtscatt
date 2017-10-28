# -*- coding: utf-8 -*-
"""
Module with some of the physical constants used to calculate all kinds of
properties used in other modules from this package.

@author: Luiz Felipe Machado Votto
"""

import numpy as np

EPSILON = 0
AXICON = np.longdouble(np.pi / 180)  # 1 degree
# np.longdouble(0.349066)  # 20 degrees
MAX_IT = 200
WAVELENGTH = np.longdouble(1064.0E-9)
WAVE_NUMBER = 2 * np.pi / WAVELENGTH
