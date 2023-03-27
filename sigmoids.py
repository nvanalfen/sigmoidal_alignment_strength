import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import lines
import streamlit as st

# Really all the same function
# redshift_sigmoid affects the y_max of color_sigmoid
# color_sigmoid affects the y_max of stellar_mass_sigmoid

def sigmoid(x, x0=0, k=1, y_low=-1, y_high=1):
    return y_low + ( y_high - y_low ) / ( 1 + np.exp( -k * (x - x0) ) )

def redshift_sigmoid(x=0, x0=0, k=1, y_low=-1, y_high=1):
    return y_low + ( y_high - y_low ) / ( 1 + np.exp( -k * (x - x0) ) )

def color_sigmoid(x=0, x0=0, k=1, y_low=0, y_high=redshift_sigmoid, params={}):
    return y_low + ( y_high(**params) - y_low ) / ( 1 + np.exp( -k * (x - x0) ) )

def stellar_mass_sigmoid(x, x0=0.5, k=1, y_low=0, y_high=color_sigmoid, params={}):
    return y_low + ( y_high(**params) - y_low ) / ( 1 + np.exp( -k * (x - x0) ) )