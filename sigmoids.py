"""
"""
import numpy as np
from collections import OrderedDict

DEFAULT_PDICT = OrderedDict()
# Fill default values and set the necessary keys
# Redhsift values
DEFAULT_PDICT["redshift_x0"] = 3
DEFAULT_PDICT["redshift_k"] = -1
DEFAULT_PDICT["redshift_y_low"] = -1
DEFAULT_PDICT["redshift_y_high"] = 1
# Color values
DEFAULT_PDICT["color_x0"] = 0
DEFAULT_PDICT["color_k"] = 1
DEFAULT_PDICT["color_y_low"] = 0
# Log stellar mass values
DEFAULT_PDICT["logsm_x0"] = 12
DEFAULT_PDICT["logsm_k"] = 1
DEFAULT_PDICT["logsm_y_low"] = 0

def sigmoid(x, x0=0, k=1, y_low=-1, y_high=1):
    return y_low + (y_high - y_low) / (1 + np.exp(-k * (x - x0)))


def redshift_sigmoid(x=0, x0=0, k=1, y_low=-1, y_high=1):
    return y_low + (y_high - y_low) / (1 + np.exp(-k * (x - x0)))


def color_sigmoid(x=0, x0=0, k=1, y_low=0, y_high=redshift_sigmoid, params={}):
    if callable(y_high):
        return y_low + (y_high(**params) - y_low) / (1 + np.exp(-k * (x - x0)))
    return y_low + (y_high - y_low) / (1 + np.exp(-k * (x - x0)))


def stellar_mass_sigmoid(x=0, x0=0.5, k=1, y_low=0, y_high=color_sigmoid, params={}):
    if callable(y_high):
        return y_low + (y_high(**params) - y_low) / (1 + np.exp(-k * (x - x0)))
    return y_low + (y_high - y_low) / (1 + np.exp(-k * (x - x0)))


def get_alignment_strength(logsm, color, redshift, **params):

    # Enforce that only parameter names are accepted as keyword args
    msg = "{0} keyword argument does not appear in DEFAULT_PDICT"
    for key in params:
        assert key in DEFAULT_PDICT.keys(), msg.format(key)

    pdict = DEFAULT_PDICT.copy()
    pdict.update(params)

    # Now call the component functions
    y_high = redshift_sigmoid( x=redshift, x0=pdict["redshift_x0"], k=pdict["redshift_k"], y_low=pdict["redshift_y_low"], y_high=pdict["redshift_y_high"] )
    y_high = color_sigmoid( x=color, x0=pdict["color_x0"], k=pdict["color_k"], y_low=pdict["color_y_low"], y_high=y_high )
    return stellar_mass_sigmoid( x=logsm, x0=pdict["logsm_x0"], k=pdict["logsm_k"], y_low=pdict["logsm_y_low"], y_high=y_high )