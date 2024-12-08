import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive, FloatSlider, VBox
# from IPython.display import display

# Function to plot the vectors
def plot_vectors(x1, y1, length1, heading1, x2, y2, length2, heading2):
    # Convert headings from degrees to radians
    heading1_rad = np.radians(heading1)
    heading2_rad = np.radians(heading2)
    
    # Calculate the x and y components of the vectors
    dx1, dy1 = length1 * np.cos(heading1_rad), length1 * np.sin(heading1_rad)
    dx2, dy2 = length2 * np.cos(heading2_rad), length2 * np.sin(heading2_rad)
    
    # Clear the current plot
    plt.figure(figsize=(6, 6))
    plt.quiver([x1, x2], [y1, y2], [dx1, dx2], [dy1, dy2], angles='xy', scale_units='xy', scale=1, color=['r', 'b'])
    
    # Set the plot limits and labels
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Interactive Vector Plot')
    plt.show()

# Define sliders for vector parameters
x1_slider = FloatSlider(value=0, min=-10, max=10, step=0.1, description='x1')
y1_slider = FloatSlider(value=0, min=-10, max=10, step=0.1, description='y1')
length1_slider = FloatSlider(value=1, min=0, max=10, step=0.1, description='Length1')
heading1_slider = FloatSlider(value=0, min=0, max=360, step=1, description='Heading1')

x2_slider = FloatSlider(value=0, min=-10, max=10, step=0.1, description='x2')
y2_slider = FloatSlider(value=0, min=-10, max=10, step=0.1, description='y2')
length2_slider = FloatSlider(value=1, min=0, max=10, step=0.1, description='Length2')
heading2_slider = FloatSlider(value=0, min=0, max=360, step=1, description='Heading2')

# Create the interactive widget
interactive_plot = interactive(
    plot_vectors,
    x1=x1_slider,
    y1=y1_slider,
    length1=length1_slider,
    heading1=heading1_slider,
    x2=x2_slider,
    y2=y2_slider,
    length2=length2_slider,
    heading2=heading2_slider
)

# Display the sliders and plot
ui = VBox(interactive_plot.children[:-1])
output = interactive_plot.children[-1]
display(ui, output)
