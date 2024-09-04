from minisom import MiniSom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import matplotlib.cm as cm
import networkx as nx
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler
import winsound
def generate_label(data, label_names):
    """
    Process the data by removing the 9th and 10th columns if present,
    and label the data based on the values in the last column using the provided label names.

    Parameters:
        data (numpy.ndarray): The input data array.
        label_names (dict): A dictionary mapping label values to label names.

    Returns:
        numpy.ndarray: The processed data array with anomalies removed.
        numpy.ndarray: An empty array (since we no longer use the 8th and 9th columns).
        numpy.ndarray: The array containing labels assigned based on the values in the last column.
    """
    # Check if the last column is present
    if data.shape[1] > 0:
        # Extract the last column
        last_column = data[:, -1]

        # Remove the last column from the original data
        data = np.delete(data, -1, axis=1)

        # Label the data based on the values in the last column using the provided label names
        labels = np.where(last_column == 0, label_names[0], label_names[1])

        print("Assigned labels based on the values in the last column.")
    else:
        print("No data found to process.")
        labels = np.array([])  # Assign an empty array to labels

    return data, labels



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


# def generate_graph_data(data, original_data, num_points_per_graph, som):
#     # Get winner neuron coordinates for all samples
#     winners = [som.winner(sample) for sample in data]
#     winner_coordinates_ls = winners
#
#     # Divide data points into sections for graph data
#     num_graphs = len(data) // num_points_per_graph
#
#
#
#     graph_data = []
#     for i in range(num_graphs):
#         # Get the data point index range of the current graph
#         start_idx = i * num_points_per_graph
#         end_idx = start_idx + num_points_per_graph
#
#         # Extract node features of the current graph
#         node_features = torch.tensor(data[start_idx:end_idx], dtype=torch.float)
#
#         # Get winner coordinate and store it
#         # Extract specified range of winner neuron coordinates based on slicing
#         winner_coordinates_list = winner_coordinates_ls[start_idx:end_idx]
#
#         # Generate edge index and edge features
#         edge_index, edge_features = generate_edges(data[start_idx:end_idx], winner_coordinates_list)
#
#         # # Get the graph label
#         # anomaly_count = np.sum(original_data[start_idx:end_idx, -2] == 1)
#         # graph_target = torch.tensor([1 if anomaly_count > num_points_per_graph // 2 else 0], dtype=torch.float).squeeze()
#         node_target = original_data[start_idx:end_idx, -1]
#         graph_data_or = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features,
#                              y=torch.tensor(node_target.astype(np.int64)))
#         graph_data.append(graph_data_or)
#     # Create a single Data object to hold all data
#
#     return graph_data,winner_coordinates_ls

def process_data(data, x):
    """
    Process the sensor data according to the specified steps.
    
    Parameters:
    - data: A list of lists, where each inner list represents the sensor data at a given timestamp.
            The first column is the timestamp, the last column is 'normal' or 'attack', and
            the other columns are sensor values.
    - x: The number of rows to process (excluding header and blank rows).
    
    Returns:
    - A numpy array of the processed data.
    """
    # Convert the data list to a DataFrame
    df = pd.DataFrame(data)
    
    # Remove the first blank row and use the second row as headers
    df.columns = df.iloc[1]
    df = df[2:]  # Data starts from the third row
    
    # Select the first x rows of actual data
    df = df.head(x)
    
    # Remove the first column (timestamps)
    df = df.iloc[:, 1:]
    
    # Replace 'normal' with 0 and 'attack' with 1 in the last column
    df.iloc[:, -1] = df.iloc[:, -1].map({'Normal': 0, 'Attack': 1})
    
    # Normalize all data (excluding the last column)
    scaler = MinMaxScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
    
    # Convert the DataFrame to a numpy array
    return df.to_numpy()




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
        anomaly_count = np.sum(original_data[start_idx:end_idx, -1] == 1)
        # print(original_data[start_idx:end_idx, -1])
        graph_target = torch.tensor([1 if anomaly_count > num_points_per_graph // 2 else 0], dtype=torch.float).squeeze()
        node_target = original_data[start_idx:end_idx, -1]
        graph_data_or = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features,
                             y=graph_target)
        graph_data_or.y2 = torch.tensor(node_target.astype(np.int64))
        graph_data.append(graph_data_or)
    # Create a single Data object to hold all data

    return graph_data,winner_coordinates_ls

def process_csv_files_and_generate_graphs(file_paths, som_shape, num_iterations,num_points_per_graph,num_data_used):
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
    # first_file_data = stats_dfs(file_paths[0]).values
    # combined_data = np.empty((0, 8))    #length of the data we need
    # combined_original_data = np.empty((0, 11)) #   with all label

    combined_data = np.empty((0, 51))
    combined_original_data = np.empty((0, 53))
    for file_path in file_paths:
        df = pd.read_csv(file_path) # Read the CSV file
        original_data = df.values
        combined_original_data = np.concatenate((combined_original_data, original_data))

    data = combined_original_data
    data = process_data(data, num_data_used)
    data, labels = generate_label(data, {0: 'Normal', 1: 'Anomaly'})
    combined_data = data.astype(np.float64)
    combined_original_data = combined_original_data[2:]
    # Train SOM
    som = MiniSom(som_shape[0], som_shape[1], combined_data.shape[1], sigma=.5, learning_rate=.5,
                  neighborhood_function='gaussian', random_seed=10)
    som.train_batch(combined_data, num_iterations, verbose=True)
    # plot_som(som, combined_data, som_shape, labels)
    # Generate graph data
    # num_points_per_graph = 10

    graph_data,winner_coordinates_ls = generate_graph_data(combined_data, combined_original_data, num_points_per_graph, som)

    return graph_data,winner_coordinates_ls


som_shape = (8, 8)
num_iterations = 100000
file_paths = ["C:/Users/lenovo/Desktop/project/degree/SWaT_Dataset_Attack.csv"

              ]
# npy format networkx temporal mrgcn batch size
num_points_per_graph=10
num_data_used=449920   #all


graph_data,node_postions = process_csv_files_and_generate_graphs(file_paths, som_shape, num_iterations,num_points_per_graph,num_data_used)


with open('swat_graphdata.pkl', 'wb') as file:
    pickle.dump(graph_data, file)
winsound.Beep(1000, 1000)