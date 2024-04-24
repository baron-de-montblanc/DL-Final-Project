import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import networkx as nx
from sklearn.metrics import confusion_matrix
from helper import human_feature, features_by_attribute


# ---------------------- Visualize input and/or output ---------------------------------



def plot_1D_distributions(data, labels, features, nbins):
    """
    Plot histograms of our data 
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




def plot_loss(loss_list, val_loss_list):

    # Visualize loss progression
    plt.figure(figsize=(6,4))
    plt.plot(loss_list, label="Training loss")
    plt.plot(val_loss_list, label="Validation loss")

    plt.title("BCE Loss as a Function of Epoch", y=1.02)
    plt.xlabel("Epoch number")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_accuracy(acc_list, val_acc_list):
    plt.figure(figsize=(6,4))
    plt.plot(acc_list, label="Training accuracy")
    plt.plot(val_acc_list, label="Validation accuracy")

    plt.title("Binary Accuracy as a Function of Epoch", y=1.02)
    plt.xlabel("Epoch number")
    plt.ylabel("Binary Accuracy")
    plt.legend()
    plt.show()


def plot_confusion_matrices(labels, preds, model_name):
    """Adapted from the following hands-on session: Lecture6_DNNClass_v0.ipynb"""
    conf_matrix = confusion_matrix(labels, preds)
    cmap = sns.color_palette("light:b", as_cmap=True)
    plt.figure(figsize=(6,4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=cmap)
    plt.title(f"{model_name} Confusion Matrix (Test Set)", y=1.02)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()



def visualize_graph(graph, x_axis='fjet_clus_eta', y_axis='fjet_clus_phi'):
    """
    Visualize the graphs that will be passed to the GNN

    Adapted from the PHYS 2550 Hands-On Session for Lecture 21
    """
    # Which features are we plotting? Compare x_axis and y_axis to hard-coded dict
    mapping = {
        'fjet_clus_pt':0,
        'fjet_clus_eta':1,
        'fjet_clus_phi':2,
        'fjet_clus_E':3,
    }
    ind1 = mapping[x_axis]
    ind2 = mapping[y_axis]

    # Convert to CPU for visualization if it's on GPU
    edge_index = graph.edge_index.cpu().numpy()
    pos = {i: [graph.x[i, ind1].item(), graph.x[i, ind2].item()] for i in range(graph.num_nodes)}

    # Create the networkx graph
    G = nx.Graph()

    # Add nodes
    for i in range(graph.num_nodes):
        G.add_node(i)

    # Add kNN edges with a specific attribute
    for start, end in edge_index.T:
        G.add_edge(int(start), int(end), edge_type='knn')

    # Generate more edges here if necessary, and add them to the graph
    # Each of these additional edges should have the edge_type attribute set to 'non_knn'
    # G.add_edge(node1, node2, edge_type='non_knn')

    # Draw the graph
    plt.figure(figsize=(6,4))
    knn_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['edge_type'] == 'knn']
    non_knn_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['edge_type'] == 'non_knn']

    # Aesthetics
    node_color = "#303030"
    edge_color_knn = sns.color_palette()[2]
    edge_color_nknn = sns.color_palette()[0]

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=40, node_color=node_color)

    # Draw kNN edges
    nx.draw_networkx_edges(G, pos, edgelist=knn_edges, edge_color=edge_color_knn, width=1.5, label='kNN Edges')

    # Draw non kNN edges
    nx.draw_networkx_edges(G, pos, edgelist=non_knn_edges, edge_color=edge_color_nknn, width=1, style='dashed', label='Non-kNN Edges')

    plt.title('Graph Visualization with kNN Edges')
    plt.legend()
    plt.xlabel(human_feature(x_axis))
    plt.ylabel(human_feature(y_axis))
    plt.show()