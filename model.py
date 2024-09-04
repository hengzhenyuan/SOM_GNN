

import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_mean_pool





# class RGCNmodel(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
#         super(RGCNmodel, self).__init__()
#         self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
#         self.conv2 = RGCNConv(hidden_channels, out_channels, num_relations
#                           )
#
#     def forward(self, data):
#         x, edge_index, edge_type = data.x, data.edge_index,data.edge_attr.to(torch.int64).squeeze()
#         batch= data.batch
#         x = self.conv1(x, edge_index, edge_type)
#         # x = F.relu(x)
#         x = self.conv2(x, edge_index, edge_type)
#
#         x = global_mean_pool(x,batch)
#
#         return x


# class RGCNmodel(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
#         super(RGCNmodel, self).__init__()
#         self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
#         self.conv2 = RGCNConv(hidden_channels, out_channels, num_relations)
#
#     def forward(self, data):
#         x, edge_index, edge_type = data.x, data.edge_index, data.edge_attr.to(torch.int64).squeeze()
#         batch = data.batch
#
#         # First convolutional layer
#         x = self.conv1(x, edge_index, edge_type)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.5, training=self.training)
#
#         # Second convolutional layer
#         node_features = self.conv2(x, edge_index, edge_type)
#
#         # Global mean pooling for graph-level representation
#         graph_representation = global_mean_pool(node_features, batch)
#
#         return node_features, graph_representation
#
#

class RGCNmodel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super(RGCNmodel, self).__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, out_channels, num_relations)

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_attr.to(torch.int64).squeeze()
        batch = data.batch

        # First convolutional layer
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Second convolutional layer
        node_features = self.conv2(x, edge_index, edge_type)

        # Apply ReLU on the output of the second convolutional layer
        node_features = F.relu(node_features)

        # Apply Softmax on the node features
        node_output = F.softmax(node_features, dim=1)

        # Global mean pooling for graph-level representation
        graph_representation = global_mean_pool(node_features, batch)

        graph_output= F.softmax(graph_representation, dim=1)
        return node_output, graph_output
