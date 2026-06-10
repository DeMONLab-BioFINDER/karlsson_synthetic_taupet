import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def get_cmap_pet():
    """
    Create a custom colormap for PET image visualization.

    Constructs a custom colormap that transitions from black (for zero/background
    values) through green and teal (low-to-mid intensity) to yellow and orange
    (high intensity). The colormap is designed to emphasize low signal values
    while providing clear contrast for high tau-PET uptake regions.
    
    Parameters
    ----------
    None

    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap
        Custom colormap object that can be used with matplotlib plotting functions via
        the ``cmap`` parameter.
    """
    # Sample points from the Viridis colormap
    viridis = plt.get_cmap('viridis')
    hot = plt.get_cmap('hot')

    # Interpolation points (values between 0 and 1)
    stops = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors = [viridis(s) for s in stops]

    # Replace the first color (dark purple) with black
    colors[0] = (0, 0, 0, 1)  # RGBA black

    # Create the list for LinearSegmentedColormap
    custom_colors = list(zip(stops, colors))

    # Make the custom colormap
    custom_colors = [
        (0.0, "black"),   # 0
        (0.15, viridis(0.25)), # dark green / teal
        (0.3, viridis(0.63)), 
        (0.45, viridis(1.0)),
        (1, hot(0.3))# max value
         ]

    cmap = LinearSegmentedColormap.from_list("blackgreen", custom_colors, N=256)
    
    return cmap