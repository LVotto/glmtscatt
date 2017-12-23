# -*- coding: utf-8 -*-
"""
Module with some useful functions that will be used all around the rest of
the code for varied purposes.

@author: Luiz Felipe Machado Votto
"""

import matplotlib.pyplot as plt
import pathlib
import pickle
import time
import winsound
import numpy as np

from glmt.constants import WAVE_NUMBER


def open_file(path='../pickles/', file_name='', formatting='.pickle', operation='rb'):
    """ Return a file with specific format either for
      reading (operation='rb') or writing (operation='wb')
    """
    return  open(str(pathlib.Path(path + file_name + formatting).absolute()), operation)

def protected_denominator(value, epsilon=np.longdouble(1E-25)):
    """ Changes a value that's zero to a small number. """
    if value == 0:
        return epsilon
    else:
        return value

def get_max_it(x_max, wave_number_k=WAVE_NUMBER):
    """ Calculates stop iteration number """
    if np.isnan(x_max):
        return 2
    return int(np.ceil(wave_number_k * x_max + np.longdouble(4.05) \
                   * pow(wave_number_k * x_max, 1/3)) + 2)

def zero(*args, **kwargs):
    """ The zero constant function """
    return 0

def one(*args, **kwargs):
    """ The function that is constant in 1 """
    return 1

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

def success_tone():
    """ Tone to call user back to the computer """
    winsound.Beep(880, 200)
    time.sleep(0.05)
    winsound.Beep(1108, 100)
    time.sleep(0.02)
    winsound.Beep(880, 100)
    time.sleep(0.02)
    winsound.Beep(1320, 1000)    

class Pickler():
    path = None
    data = None
    
    def __init__(self, path=None, file_name=None, data=None):
        if file_name:
            self.path = '../pickles/' + file_name        
        if path:
            self.path = path
        if data:
            self.data = data
            self.store()
            
    def read(self):
        if not self.data:
            with open(str(pathlib.Path(self.path).absolute()), 'rb') as f:
                self.data = pickle.load(f)
        return self.data
    
    def store(self):
        with open(str(pathlib.Path(self.path).absolute()), 'wb') as f:
            pickle.dump(self.data, f)            
