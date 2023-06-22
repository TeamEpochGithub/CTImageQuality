import numpy as np
from matplotlib import pyplot as plt


def plot_dist(labels):
    unique_values, value_counts = np.unique(labels, return_counts=True)

    fig, ax = plt.subplots()
    ax.bar(unique_values, value_counts, width=0.2)  # Set the width of the bars to 0.2

    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')
    ax.set_title('Discrete Value Distribution')
    ax.set_xticks(np.arange(0, 4.2, 0.2))  # Set the x-axis ticks from 0 to 4 with a step of 0.2

    plt.show()