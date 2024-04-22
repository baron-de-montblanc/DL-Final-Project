import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from helper import human_feature, features_by_attribute


# ---------------------- Visualize input and/or output ---------------------------------



def plot_1D_distributions(data, labels, features, nbins):
    """
    Plot histograms of our data given the desired attribute (from 'jet' or 'constituents')
    """

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    # Dynamic range of x depending on feature (hard-coded)
    if 'fjet_clus_pt' not in features:
        xmin = [0,-2.5,-4,0]
        xmax = [5e6,2.5,4,5e5]
    else:
        xmin = [0,-4,-4,0]
        xmax = [5,4,4,5]

    xlabels = ["MeV", "rad", "rad", "MeV"]

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

            d_one = d[one_labels].flatten()
            d_zero = d[zero_labels].flatten()

            # Get rid of zeros (since they most likely just correspond to zero-padded elements)
            d_one = d_one[d_one != 0]
            d_zero = d_zero[d_zero != 0]

            # Plotting
            axes[i,j].hist(d_one, bins=nbins, density=True, alpha=0.8, color=c1, range=(xmin[f_idx],xmax[f_idx]))
            axes[i,j].hist(d_zero, bins=nbins, density=True, alpha=0.8, color=c2, range=(xmin[f_idx],xmax[f_idx]))
            axes[i,j].set_title(f'Distribution of {human_feature(f)} by label')
            axes[i,j].set_xlabel(f"{human_feature(f)} ({xlabels[f_idx]})")
            axes[i,j].set_ylabel('Density')
            axes[i,j].legend(title='Label', labels=['Top Quark (signal)', 'Background'])

    plt.tight_layout()
    plt.show()