from minisom import MiniSom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import matplotlib.cm as cm
import networkx as nx
import torch
from torch_geometric.data import Data

def stats_dfs(path):
    df = pd.read_csv(path, sep=";")
    print("\n_________________\n")
    print(path)
    print("\n_________________\n")
    print(df.shape)
    print("\n_________________\n")
    return df


def find_anomalies(data):
    num_exceptions = 0
    exception_timestamps = []

    for row in data:
        timestamp, is_exception = row[0], row[9]
        if is_exception == 1:
            num_exceptions += 1
            exception_timestamps.append(timestamp)

    return num_exceptions, exception_timestamps


def find_change_point(data):
    num_exceptions = 0
    exception_timestamps = []

    for row in data:
        timestamp, is_exception = row[0], row[10]
        if is_exception == 1:
            num_exceptions += 1
            exception_timestamps.append(timestamp)

    return num_exceptions, exception_timestamps


# normalize the data
def normalize_data(data):
    numerical_data = data[:, 1:].astype(float)
    mean = np.mean(numerical_data, axis=0)
    std = np.std(numerical_data, axis=0)
    normalized_data = np.hstack((data[:, :1], (numerical_data - mean) / std))
    return normalized_data


def remove_first_column(data):
    """
    Remove the first column of the array.

    Parameters:
        data (numpy.ndarray): The input array.

    Returns:
        numpy.ndarray: The array with the first column removed.
    """
    return data[:, 1:]


def process_data(data, label_names):
    """
    Process the data by removing the 9th and 10th columns if present,
    and label the data based on the values in the 9th and 10th columns using the provided label names.

    Parameters:
        data (numpy.ndarray): The input data array.
        label_names (dict): A dictionary mapping label values to label names.

    Returns:
        numpy.ndarray: The processed data array with anomalies removed.
        numpy.ndarray: The array containing the 8th and 9th columns if present, otherwise empty.
        numpy.ndarray: The array containing labels assigned based on the values in the 9th and 10th columns.
    """
    # Check if the 8th and 9th columns are present
    if data.shape[1] >= 9:
        # Extract the 8th and 9th columns (from 0 th columns)
        col_8_9 = data[:, 8:10]

        # Remove the 8th and 9th columns from the original data
        data = np.delete(data, [8, 9], axis=1)

        # Label the data based on the values in the 8th and 9th columns using the provided label names
        labels = np.where((col_8_9[:, 0] == 0) & (col_8_9[:, 1] == 0), label_names[1],  # Normal
                          np.where((col_8_9[:, 0] == 1) & (col_8_9[:, 1] == 0), label_names[2],  # Anomaly
                                   np.where((col_8_9[:, 0] == 0) & (col_8_9[:, 1] == 1), label_names[3],  # Changepoint
                                            label_names[3])))  # Changepoint

        print("Removed anomalies from the 8th and 9th columns.")
    else:
        print("No anomalies found in the 8th and 9th columns.")
        col_8_9 = np.array([])  # Assign an empty array to col_8_9
        labels = np.array([])  # Assign an empty array to labels

    return data, col_8_9, labels


def plot_features(data, num_points=500):
    """
    Plot the features for the  data points.

    Parameters:
        data (numpy.ndarray): The data array containing time and feature values.
        num_points (int): Number of data points to plot (default is 500).
    """
    # Replace the first column with an incrementing sequence
    data[:, 0] = np.arange(len(data))

    # Print the first few rows of data for verification
    print("First few rows of data for verification:")
    print(data[:5])
    print("Shape of data:", data.shape)
    print("Data type:", type(data))

    # Create subplots for each feature
    fig, axs = plt.subplots(8, 1, figsize=(10, 20), sharex=True)

    # Plot each feature
    for i, ax in enumerate(axs):
        ax.plot(time, features[:, i])
        ax.set_title(f'Feature {i + 1}')
        ax.set_ylabel('Feature Value')

    axs[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show(block=True)


def plot_som(som_model, data, som_shape, labels):
    """
    Plot the SOM grid along with the winning neurons for the given data,
    with different colors for different labels.

    Parameters:
        som_model: The trained SOM model.
        data (numpy.ndarray): The data array used for training SOM.
        som_shape (tuple): The shape of the SOM grid.
        labels (numpy.ndarray): The labels for each data point.


    """
    # Get the coordinates of the winning neurons for each data point
    w_x, w_y = zip(*[som_model.winner(d) for d in data])
    w_x = np.array(w_x)
    w_y = np.array(w_y)

    # Calculate the figure size based on the SOM grid shape
    som_width, som_height = som_shape
    fig_width = som_width  # Adjust the multiplier as needed
    fig_height = som_height  # Adjust the multiplier as needed

    # Plot the SOM grid
    plt.figure(figsize=(fig_width, fig_height))
    plt.pcolor(som_model.distance_map().T, cmap='bone_r', alpha=.2)
    plt.colorbar()

    # Plot the winning neurons with different colors for different labels
    unique_labels = np.unique(labels)
    colors = cm.tab10(np.linspace(0, 1, len(unique_labels)))  # Generate colors based on the number of unique labels
    for i, label in enumerate(unique_labels):
        idx = np.where(labels == label)[0]
        plt.scatter(w_x[idx] + .5 + (np.random.rand(len(idx)) - .5) * .8,
                    w_y[idx] + .5 + (np.random.rand(len(idx)) - .5) * .8,
                    s=50, color=colors[i], label=label)

    plt.legend(loc='upper right')
    plt.grid()
    plt.show(block=True)




def generate_edges(data, winner_coordinates):
    # time edge
    time_edges = [[i, i + 1] for i in range(len(data) - 1)]

    spatial_edges = [[i, j] for i in range(len(data)) for j in range(len(data))
                     if i != j and np.all(winner_coordinates[i] == winner_coordinates[j])]

    # Merging edges in time and space
    edges = time_edges + spatial_edges

    # Edge feature matrix
    edge_labels = []
    for edge in edges:
        if edge in time_edges:
            edge_labels.append([0])
        else:
            edge_labels.append([1])

    return torch.tensor(edges, dtype=torch.long).t().contiguous(), torch.tensor(edge_labels, dtype=torch.float)


def generate_graph_data(data, original_data, num_points_per_graph, som):
    # Get winner neuron coordinates for all samples
    winners = [som.winner(sample) for sample in data]
    winner_coordinates_ls = winners

    # Divide data points into sections for graph data
    num_graphs = len(data) // num_points_per_graph



    graph_data = []
    for i in range(num_graphs):
        # Get the data point index range of the current graph
        start_idx = i * num_points_per_graph
        end_idx = start_idx + num_points_per_graph

        # Extract node features of the current graph
        node_features = torch.tensor(data[start_idx:end_idx], dtype=torch.float)

        # Get winner coordinate and store it
        # Extract specified range of winner neuron coordinates based on slicing
        winner_coordinates_list = winner_coordinates_ls[start_idx:end_idx]

        # Generate edge index and edge features
        edge_index, edge_features = generate_edges(data[start_idx:end_idx], winner_coordinates_list)

        # Get the graph label
        anomaly_count = np.sum(original_data[start_idx:end_idx, -2] == 1)
        graph_target = torch.tensor([1 if anomaly_count > num_points_per_graph // 2 else 0], dtype=torch.float).squeeze()
        node_target = original_data[start_idx:end_idx, -2]
        graph_data_or = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features,
                             y=graph_target)
        graph_data_or.y2 =torch.tensor(node_target.astype(np.int64))

        graph_data.append(graph_data_or)
    # Create a single Data object to hold all data

    return graph_data,winner_coordinates_ls


def process_csv_files_and_generate_graphs(file_paths, som_shape, num_iterations,num_points_per_graph):
    """
     Process multiple CSV files and generate graph data.

     Parameters:
        file_paths (list): A list of file paths to CSV files.
        som_shape (tuple): A tuple specifying the shape of the SOM grid.
        num_iterations (int): Number of iterations for SOM training.

     Returns:
        Data: A Data object containing the graph data for all CSV files combined.
    """
    # Initialize combined data with the shape of the first file
    first_file_data = stats_dfs(file_paths[0]).values
    combined_data = np.empty((0, 8)) #length of the data we need
    combined_original_data = np.empty((0, 11))#  orginal data with all label
    for file_path in file_paths:
        df = stats_dfs(file_path)
        original_data = df.values

        combined_original_data = np.concatenate((combined_original_data, original_data))

    num_exceptions, exception_timestamps = find_anomalies(combined_original_data)
    num_change_points, change_point_timestamps = find_change_point(combined_original_data)
    print("Anomalies:", num_exceptions)
    print("Change points:", num_change_points)
    data = combined_original_data
    data = remove_first_column(data)
    data, _, labels = process_data(data, {1: 'Normal', 2: 'Anomaly', 3: 'Changepoint'})
    data = normalize_data(data)
    combined_data = data.astype(np.float64)

    # Train SOM
    som = MiniSom(som_shape[0], som_shape[1], combined_data.shape[1], sigma=.5, learning_rate=.5,
                  neighborhood_function='gaussian', random_seed=10)
    som.train_batch(combined_data, num_iterations, verbose=True)
    plot_som(som, combined_data, som_shape, labels)
    # Generate graph data
    # num_points_per_graph = 10

    graph_data,winner_coordinates_ls = generate_graph_data(combined_data, combined_original_data, num_points_per_graph, som)

    return graph_data,winner_coordinates_ls




def visualize_graph(data):
    x = data.x.numpy()
    edge_index = data.edge_index.numpy()
    edge_attr = data.edge_attr.numpy() if data.edge_attr is not None else None
    y = data.y.numpy() if data.y is not None else None

    #
    print("x:", x)
    print("edge_index:", edge_index)
    print("edge_attr:", edge_attr)
    print("y:", y)

    #
    G = nx.DiGraph()
    # G = nx.grid_2d_graph(3, 3)
    # node
    for i, feature in enumerate(x):
        # label = y[i] if y is not None and i < len(y) else ''
        G.add_node(i, feature=feature[0])

    # edge
    for i in range(edge_index.shape[1]):
        source = edge_index[0, i]
        target = edge_index[1, i]
        weight = edge_attr[i, 0] if edge_attr is not None else 1.5
        G.add_edge(source, target, weight=weight)

    # color
    edge_colors = []
    if edge_attr is not None:
        for attr in edge_attr:
            if attr[0] == 0.:
                edge_colors.append('blue')
            if attr[0] == 1.5:
                edge_colors.append('red')
            elif attr[0] == 1.:
                edge_colors.append('black')
            else:
                edge_colors.append('black')  #
    else:
        edge_colors = 'black'

    #
    pos = nx.spring_layout(G)  # spring layout
    # pos = nx.planar_layout(G)
    labels = nx.get_node_attributes(G, 'feature')

    nx.draw(G, pos, with_labels=False, labels=labels, node_size=100, node_color='skyblue', font_size=1,
            font_color='black', font_weight='bold', edge_color=edge_colors)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title('Graph Visualization using networkx')
    plt.show(block=True)


# SOM
som_shape = (8, 8)
num_iterations = 10000
file_paths = [
"C:/Users/lenovo/Desktop/project/degree/archive/SKAB/valve1/0.csv",
"C:/Users/lenovo/Desktop/project/degree/archive/SKAB/valve1/1.csv",
"C:/Users/lenovo/Desktop/project/degree/archive/SKAB/valve1/2.csv",
"C:/Users/lenovo/Desktop/project/degree/archive/SKAB/valve1/3.csv",
"C:/Users/lenovo/Desktop/project/degree/archive/SKAB/valve1/4.csv",
"C:/Users/lenovo/Desktop/project/degree/archive/SKAB/valve1/5.csv",
"C:/Users/lenovo/Desktop/project/degree/archive/SKAB/valve1/6.csv",
"C:/Users/lenovo/Desktop/project/degree/archive/SKAB/valve1/7.csv",
"C:/Users/lenovo/Desktop/project/degree/archive/SKAB/valve1/8.csv",
"C:/Users/lenovo/Desktop/project/degree/archive/SKAB/valve1/9.csv",
"C:/Users/lenovo/Desktop/project/degree/archive/SKAB/valve1/10.csv",
"C:/Users/lenovo/Desktop/project/degree/archive/SKAB/valve1/11.csv",
"C:/Users/lenovo/Desktop/project/degree/archive/SKAB/valve1/12.csv",
"C:/Users/lenovo/Desktop/project/degree/archive/SKAB/valve1/13.csv",
"C:/Users/lenovo/Desktop/project/degree/archive/SKAB/valve1/14.csv",
"C:/Users/lenovo/Desktop/project/degree/archive/SKAB/valve1/15.csv",
              ]
# npy format networkx temporal mrgcn batch size
num_points_per_graph=10



graph_data,node_postions = process_csv_files_and_generate_graphs(file_paths, som_shape, num_iterations,num_points_per_graph)



# print(graph_data[0])
# visualize_graph(graph_data[0])
# visualize_graph(graph_data[20])

with open('graphdata.pkl', 'wb') as file:
    pickle.dump(graph_data, file)

# with open('graphdata.pkl', 'rb') as file:
#     loaded_graphdata = pickle.load(file)

# print(loaded_graphdata)
