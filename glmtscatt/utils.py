# -*- coding: utf-8 -*-
"""
Module with some useful functions that will be used all around the rest of
the code for varied purposes.

@author: Luiz Felipe Machado Votto
"""

import json
import matplotlib.pyplot as plt
import pathlib
import pickle
import time
import numpy as np
import base64

from glmtscatt.constants import WAVE_NUMBER


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

def get_max_it(x_max, wave_number_k=np.abs(WAVE_NUMBER)):
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
    import winsound
    """ Tone to call user back to the computer """
    winsound.Beep(880, 200)
    time.sleep(0.05)
    winsound.Beep(1108, 100)
    time.sleep(0.02)
    winsound.Beep(880, 100)
    time.sleep(0.02)
    winsound.Beep(1320, 1000)

class JSONStoreManager(json.encoder.JSONEncoder):
    """ A class that stores and retrieves json encoded data. """
    path = None
    data = None

    def __init__(self, path=None, data=None, **kwargs):
        super(JSONStoreManager, self).__init__(**kwargs)
        self.path = path
        self.data = data

    def default(self, o):
       try:
           iterable = iter(o)
       except TypeError:
           pass
       else:
           return list(iterable)
       if isinstance(o, np.float128):
           return json.JSONEncoder.default(self, np.float64(o))
       # Let the base class default method raise the TypeError
       return json.JSONEncoder.default(self, o)

    def store(self):
        with open(self.path, 'w') as f:
            return json.dump(self.encode(self.data),
                             f, sort_keys=True,
                             indent=4,
                             separators=(',', ':'),
                             allow_nan=False)


class PlotHandler(JSONStoreManager):
    shape = None
    labels = []
    title = ''
    def __init__(self, shape=None, labels=None, title=None, *args, **kwargs):
        self.shape = shape
        self.title = title
        self.labels = labels
        super(PlotHandler, self).__init__(*args, **kwargs)
        self.data = {'shape': self.shape,
                     'title': self.title,
                     'labels': self.labels,
                     'data': self.data}

    def plot(self):
        plt.figure()
        if self.shape == 1:
            plt.plot(self.data['data'][0], self.data['data'][1])
            plt.xlabel(self.data['labels'][0])
            plt.ylabel(self.data['labels'][1])
            plt.title(self.data['title'])

        if self.shape == 2:
            ax = plt.axes()
            cs = ax.contourf(self.data['data'][0],
                              self.data['data'][1],
                              self.data['data'][2])
            plt.colorbar(cs, format="%.2f")
            ax.set_aspect('equal', 'box')
            ax.set_xlabel(self.data['labels'][0])
            ax.set_ylabel(self.data['labels'][1])
            plt.title(self.data['title'])

        plt.show()


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
