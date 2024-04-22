import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from helper import human_feature, features_by_attribute


# ---------------------- Visualize input and/or output ---------------------------------



def plot_1D_distributions(data, labels, attribute):
    """
    Plot histograms of our data given the desired attribute (from 'jet' or 'constituents')
    """

    features = features_by_attribute(attribute)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    # Dynamic range of x depending on feature (hard-coded)
    xmin = [0,-2.5,-4.5,0]
    xmax = [5e6,2.5,4.5,5e5]

    # Set colors (for aesthetic purposes!)
    c1 = sns.color_palette()[2]
    c2 = sns.color_palette()[0]

    for i in range(2):
        for j in range(2):
            f_idx = int(2*i+j)

            f = features[f_idx]
            d = data[:,f_idx]

            # Separate by labels
            one_labels = labels.astype(bool)
            zero_labels = np.invert(one_labels)

            d_one = d[one_labels]
            d_zero = d[zero_labels]

            # Plotting
            axes[i,j].hist(d_one, bins='auto', density=True, alpha=0.8, color=c1)
            axes[i,j].hist(d_zero, bins='auto', density=True, alpha=0.8, color=c2)
            axes[i,j].set_title(f'Distribution of {human_feature(f)} by label')
            axes[i,j].set_xlabel(human_feature(f))
            axes[i,j].set_ylabel('Density')
            axes[i,j].set_xlim(xmin[f_idx],xmax[f_idx])
            axes[i,j].legend(title='Label', labels=['Top Quark (signal)', 'Background'])

    plt.tight_layout()
    plt.show()