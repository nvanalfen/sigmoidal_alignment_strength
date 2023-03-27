"""
"""
import numpy as np
from collections import OrderedDict

DEFAULT_PDICT = OrderedDict()


def sigmoid(x, x0=0, k=1, y_low=-1, y_high=1):
    return y_low + (y_high - y_low) / (1 + np.exp(-k * (x - x0)))


def redshift_sigmoid(x=0, x0=0, k=1, y_low=-1, y_high=1):
    return y_low + (y_high - y_low) / (1 + np.exp(-k * (x - x0)))


def color_sigmoid(x=0, x0=0, k=1, y_low=0, y_high=redshift_sigmoid, params={}):
    return y_low + (y_high(**params) - y_low) / (1 + np.exp(-k * (x - x0)))


def stellar_mass_sigmoid(x, x0=0.5, k=1, y_low=0, y_high=color_sigmoid, params={}):
    return y_low + (y_high(**params) - y_low) / (1 + np.exp(-k * (x - x0)))


def get_alignment_strength(logsm, color, redshift, **params):

    # Enforce that only parameter names are accepted as keyword args
    msg = "{0} keyword argument does not appear in DEFAULT_PDICT"
    for key in params:
        assert key in DEFAULT_PDICT.keys(), msg.format(key)

    pdict = DEFAULT_PDICT.copy()
    pdict.update(params)

    # Now call the component functions

    raise NotImplementedError()
