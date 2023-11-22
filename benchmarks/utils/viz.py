import matplotlib.pyplot as plt
import numpy as np

def scatter_plot_with_correlation_line(x, y):
    '''
    http://stackoverflow.com/a/34571821/395857
    x does not have to be ordered.
    '''
    # Create scatter plot
    plt.scatter(x, y)

    # Add correlation line
    axes = plt.gca()
    m, b = np.polyfit(x, y, 1)
    X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
    plt.plot(X_plot, m*X_plot + b, '-')

    # Save figure
    #plt.savefig(graph_filepath, dpi=300, format='png', bbox_inches='tight')