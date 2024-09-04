import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.cm as cm
from minisom import MiniSom
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler

def process_data_new(data, x):
    """
    Process the sensor data for the new dataset.
    
    Parameters:
    - data: A numpy array with the sensor data.
    - x: The number of rows to process.
    
    Returns:
    - A numpy array of processed data.
    """
    # Convert the data list to a DataFrame
    df = pd.DataFrame(data)
    
    # Select the first x rows of actual data
    df = df.head(x)
    # remove the timestamp
    df = df.iloc[:, 1:]
    # Replace '-999' with 0 and '1' with 1 in the last column
    df.iloc[:, -1] = df.iloc[:, -1].map({-999: 0, 1: 1})
    
    # Normalize all data (excluding the last column)
    scaler = MinMaxScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
    
    # Convert the DataFrame to a numpy array
    return df.to_numpy()

def generate_label(data, label_names):
    """
    Process the data by removing unnecessary columns and label the data.
    
    Parameters:
        data (numpy.ndarray): The input data array.
        label_names (dict): A dictionary mapping label values to label names.
    
    Returns:
        numpy.ndarray: The processed data array with anomalies removed.
        numpy.ndarray: An array containing labels assigned based on the values in the last column.
    """
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

def generate_graph_data_new(data, original_data, num_points_per_graph, som):
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
        winner_coordinates_list = winner_coordinates_ls[start_idx:end_idx]

        # Generate edge index and edge features
        edge_index, edge_features = generate_edges(data[start_idx:end_idx], winner_coordinates_list)

        # Get the graph label
        anomaly_count = np.sum(original_data[start_idx:end_idx, -1] == 1)
        graph_target = torch.tensor([1 if anomaly_count > num_points_per_graph // 2 else 0], dtype=torch.float).squeeze()
        node_target= original_data[start_idx:end_idx, -1]
        graph_data_or = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features,
                             y=graph_target)
        graph_data_or.y2 = torch.tensor(node_target.astype(np.int64))
        graph_data.append(graph_data_or)
    return graph_data, winner_coordinates_ls

def process_csv_files_and_generate_graphs_new(file_paths, som_shape, num_iterations, num_points_per_graph, num_data_used):
    """
    Process multiple CSV files and generate graph data for the new dataset.
    
    Parameters:
        file_paths (list): A list of file paths to CSV files.
        som_shape (tuple): A tuple specifying the shape of the SOM grid.
        num_iterations (int): Number of iterations for SOM training.
        num_points_per_graph (int): Number of points per graph.
        num_data_used (int): Number of data rows to use.
    
    Returns:
        graph_data: A list of Data objects containing the graph data for all CSV files combined.
        winner_coordinates_ls: List of winner neuron coordinates for each data point.
    """
    # Initialize combined data with the shape of the first file
    combined_data = np.empty((0, 45))  # 43 features + 1 label
    combined_original_data = np.empty((0, 45))  # Original data with label

    for file_path in file_paths:
        df = pd.read_csv(file_path)  # Read the CSV file
        original_data = df.values
        combined_original_data = np.concatenate((combined_original_data, original_data))

    data = combined_original_data
    data = process_data_new(data, num_data_used)
    data, labels = generate_label(data, {0: 'Normal', 1: 'Anomaly'})
    combined_data = data.astype(np.float64)

    # Train SOM
    som = MiniSom(som_shape[0], som_shape[1], combined_data.shape[1], sigma=.5, learning_rate=.5,
                  neighborhood_function='gaussian', random_seed=10)
    som.train_batch(combined_data, num_iterations, verbose=True)
    # plot_som(som, combined_data, som_shape, labels)

    # Generate graph data
    graph_data, winner_coordinates_ls = generate_graph_data_new(combined_data, combined_original_data, num_points_per_graph, som)

    return graph_data, winner_coordinates_ls

# Define parameters
som_shape = (8, 8)
num_iterations = 100000
file_paths = ["C:/Users/lenovo/Downloads/BATADAL_dataset04.csv"]  # Replace with actual file path
num_points_per_graph = 10
num_data_used = 449920  # Adjust based on your needs

# Process the data and generate graphs
graph_data, node_positions = process_csv_files_and_generate_graphs_new(file_paths, som_shape, num_iterations, num_points_per_graph, num_data_used)

# Save the graph data to a pickle file
with open('BAT_graphdata.pkl', 'wb') as file:
    pickle.dump(graph_data, file)
