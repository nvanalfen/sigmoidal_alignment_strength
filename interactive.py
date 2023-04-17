import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import lines
import streamlit as st
from sigmoids import stellar_mass_sigmoid, color_sigmoid, redshift_sigmoid, DEFAULT_PDICT

default_redshift_y_low = DEFAULT_PDICT["redshift_y_low"]
default_redshift_y_high = DEFAULT_PDICT["redshift_y_high"]
default_redshift_x0 = DEFAULT_PDICT["redshift_x0"]
default_redshfit_k = DEFAULT_PDICT["redshift_k"]

default_color_y_low = DEFAULT_PDICT["color_y_low"]
default_color_x0 = DEFAULT_PDICT["color_x0"]
default_color_k = DEFAULT_PDICT["color_k"]

default_logsm_y_low = DEFAULT_PDICT["logsm_y_low"]
default_logsm_x0 = DEFAULT_PDICT["logsm_x0"]
default_logsm_k = DEFAULT_PDICT["logsm_k"]

# Sliders and Titles
st.title("Alignment Strengths")

st.sidebar.title("Chart Values")
grid_dimension = st.sidebar.slider("Subplots Dimension", 1, 3, 2, step=1)
min_red, max_red = st.sidebar.slider("Redshift View Range", -20., 20., (-2., 5.), step=0.1)
min_color, max_color = st.sidebar.slider("Color View Range", -10., 10., (-5., 5.), step=0.1)
min_m, max_m = st.sidebar.slider("Log Stellar Mass Range", 0., 20., (0., 15.), step=0.1)

st.sidebar.title("Redshift Parameters")
redshift_x0 = st.sidebar.slider("Redshift x0", 0., 20., default_redshift_x0, step=0.1)
redshift_k = st.sidebar.slider("Redshift k", -10., 10., default_redshfit_k, step=0.1)
redshift_y_low = st.sidebar.slider("Redshift y low", -1., 1., default_redshift_y_low, step=0.1)
redshift_y_high = st.sidebar.slider("Redshift y high", 0., 2., default_redshift_y_high, step=0.1)

st.sidebar.title("Color Parameters")
color_x0 = st.sidebar.slider("Color x0", -5., 5., default_color_x0, step=0.1)
color_k = st.sidebar.slider("Color k", -10., 10., default_color_k, step=0.1)
color_y_low = st.sidebar.slider("Color y low", -1., 1., default_color_y_low, step=0.1)

st.sidebar.title("Mass Parameters")
mass_x0 = st.sidebar.slider("Mass x0", -10., 20., default_logsm_x0, step=0.1)
mass_k = st.sidebar.slider("Mass k", -10., 10., default_logsm_k, step=0.1)
mass_y_low = st.sidebar.slider("Mass y low", -1., 1., default_logsm_y_low, step=0.1)

# Parameter Setting
base_redshift_params = {"x0":redshift_x0, "k":redshift_k, "y_low":redshift_y_low, "y_high":redshift_y_high}
base_color_params = {"x0":color_x0, "k":color_k, "y_low":color_y_low, "y_high":redshift_sigmoid}
base_mass_params = {"x0":mass_x0, "k":mass_k, "y_low":mass_y_low, "y_high":color_sigmoid}

mred = u'#d62728' 
mgreen = u'#2ca02c'
mblue = u'#1f77b4'

redshift_steps_size = 0.5
redshifts = np.arange(0., redshift_steps_size*grid_dimension*grid_dimension + redshift_steps_size, redshift_steps_size)
color_indices = np.arange(-5, 5.2, 0.2)
logM = np.arange(min_m-1, max_m+1, 0.1)

# Color lines appropriately
colors=cm.coolwarm(np.linspace(0,1,len(color_indices))) # blue first

red_line=lines.Line2D([],[],ls='-',c=mred,label="Red (color index = {})".format(max(color_indices).round(1)) )
blue_line=lines.Line2D([],[],ls='-',c=mblue,label="Blue (color index = {})".format(min(color_indices).round(1)) )

# Plots

# Plot of Three separate Sigmoids
st.title("Base Sigmoids")
fig, axes = plt.subplots(1, 3, figsize=(20,6))

red_range = np.arange(min_red, max_red+0.1, 0.1)
axes[0].set_title("Redshift", fontsize=25)
axes[0].set_xlabel("Redshift", fontsize=25)
axes[0].set_ylabel("Color y_high", fontsize=25)
axes[0].plot( red_range, redshift_sigmoid(red_range, **base_redshift_params) )
axes[0].set_xlim([min_red, max_red])
axes[0].set_ylim([-0.1, 1.1])
axes[0].tick_params(labelsize=20)

params = { **base_color_params, "params":{**base_redshift_params, "x":-np.inf} }
color_range = np.arange(min_color, max_color+0.1, 0.1)
axes[1].set_title("Color", fontsize=25)
axes[1].set_xlabel("Color Index", fontsize=25)
axes[1].set_ylabel("Mass y_high", fontsize=25)
axes[1].plot( color_range, color_sigmoid(color_range, **params ) )
axes[1].set_xlim([min_color, max_color])
axes[1].set_ylim([-0.1, 1.1])
axes[1].tick_params(labelsize=20)

params = { **base_mass_params, "params":{**params, "x":np.inf} }
mass_range = np.arange(min_m, max_m+0.1, 0.1)
axes[2].set_title("Alignment Strength", fontsize=25)
axes[2].set_xlabel("Log(M*)", fontsize=25)
axes[2].set_ylabel("Alignment Strength", fontsize=25)
axes[2].plot( mass_range, stellar_mass_sigmoid(mass_range, **params ) )
axes[2].set_xlim([min_m, max_m])
axes[2].set_ylim([-0.1, 1.1])
axes[2].tick_params(labelsize=20)

plt.tight_layout()

st_fig = st.pyplot(fig=fig)

# Subplot Grid
st.title("Alignment Evolution")
fig, axes = plt.subplots(grid_dimension, grid_dimension, figsize=(20,15), sharex=True, sharey=True)

axes = np.atleast_1d(axes)

# Loop through each redshift
# Each redshift gets its own subplot
for i, ax in enumerate( axes.flatten() ):
    # Set up the parameters common for this redshift value
    redshift = redshifts[i]
    new_color_params = { **base_color_params, "params":{**base_redshift_params, "x":redshift} }
    
    # Prepare plot labels
    ax.set_title("Redshift = {}".format(redshift), fontsize=25)
    ax.set_ylabel("Alignment Strength", fontsize=25)
    ax.set_xlabel("Log($M_*$)", fontsize=25)
    ax.tick_params(labelsize=20)
    
    # Loop through each color
    # Each redshift plot will show multiple color indices
    for j, color_index in enumerate(color_indices):
        new_mass_params = { **base_mass_params, "params":{**new_color_params, "x":color_index} }
        ax.plot( logM, stellar_mass_sigmoid( logM, **new_mass_params ), lw=0.5, color=colors[j] )
    
    # Draw a black horizontal line at the maximum possible alignment value for the reddest (most aligned) galaxies
    asymptote = stellar_mass_sigmoid( np.inf, **new_mass_params )
    ax.axhline(y=asymptote, xmin=0, xmax=max_m+1, linewidth=0.5, color='k')
    
# Set the color legend
axes.flatten()[-1].legend(handles=[red_line, blue_line])
plt.tight_layout()

plt.xlim([min_m, max_m])
plt.ylim([0,1])

#plt.show()

st_fig = st.pyplot(fig=fig, figsize=(20,15))