import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Define the colormap using LinearSegmentedColormap
cdict = {
    'red':   [(0.0, 1.0, 1.0),    # Red at the start
              (0.5, 1.0, 1.0),    # White in the middle
              (1.0, 0.0, 0.0)],   # Blue at the end
    
    'green': [(0.0, 0.0, 0.0),    # Red at the start
              (0.5, 1.0, 1.0),    # White in the middle
              (1.0, 0.0, 0.0)],   # Blue at the end
    
    'blue':  [(0.0, 0.0, 0.0),    # Red at the start
              (0.5, 1.0, 1.0),    # White in the middle
              (1.0, 1.0, 1.0)]    # Blue at the end
}

custom_cmap = mcolors.LinearSegmentedColormap('RedWhiteBlue', cdict)

# Create some data for the contour plot
x = np.linspace(-3.0, 3.0, 100)
y = np.linspace(-3.0, 3.0, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

# Plot the contour plot
fig, ax = plt.subplots()
contour = ax.contourf(X, Y, Z, 20, cmap=custom_cmap)

# Create a divider for the existing axes instance
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="30%", pad=0.05)

# Create the colorbar in the new axes
cbar = fig.colorbar(contour, cax=cax)

# Adjust the plot layout to make space for the colorbar
plt.subplots_adjust(right=0.85)

# Add title and labels
plt.title('Custom Colormap - Red, White, Blue')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

plt.show()
