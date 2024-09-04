import pickle
from torch_geometric.loader import DataLoader
import torch
from torch.optim import Adam
from sklearn.metrics import f1_score
from model import RGCNmodel
import winsound
import csv

with open('BAT_graphdata.pkl', 'rb') as file:
    graph_data = pickle.load(file)



train_size = int(0.8 * len(graph_data))
test_size = len(graph_data) - train_size
train_data, test_data = torch.utils.data.random_split(graph_data, [train_size, test_size])
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
test_loader = DataLoader(test_data, batch_size=4, shuffle=True)



in_channels = 43
hidden_channels = 16
out_channels = 2
num_relations = 2


model = RGCNmodel(in_channels, hidden_channels, out_channels, num_relations)
optimizer = Adam(model.parameters(), lr=0.001)

criterion = torch.nn.CrossEntropyLoss()#


# Training function
def train(model, train_loader, optimizer, criterion):
    model.train()
    epoch_loss = 0.0
    for data in train_loader:
        optimizer.zero_grad()
        node_features, graph_representation = model(data)
        loss = criterion(graph_representation, data.y.long()) + criterion(node_features,
                                                                          data.y2)  # Assuming node-level labels in data.y2
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_epoch_loss = epoch_loss / len(train_loader)
    return avg_epoch_loss


# Testing function
from sklearn.metrics import precision_score, recall_score, f1_score


def test(model, test_loader, criterion):
    model.eval()
    correct_graph = 0
    correct_node = 0
    test_loss = 0
    total = 0
    graph_predictions = []
    graph_targets = []
    node_predictions = []
    node_targets = []

    with torch.no_grad():
        for data in test_loader:
            node_features, graph_representation = model(data)
            loss = criterion(graph_representation, data.y.long()) + criterion(node_features, data.y2.long())
            test_loss += loss.item()
            total += data.y2.size(0)

            # Graph-level predictions
            graph_pred = graph_representation.argmax(dim=1)
            correct_graph += int((graph_pred == data.y).sum())
            graph_predictions.extend(graph_pred.tolist())
            graph_targets.extend(data.y.tolist())

            # Node-level predictions
            node_pred = node_features.argmax(dim=1)
            correct_node += int((node_pred == data.y2).sum())
            node_predictions.extend(node_pred.tolist())
            node_targets.extend(data.y2.tolist())

    graph_accuracy = correct_graph / len(test_loader.dataset)
    node_accuracy = correct_node / total
    avg_test_loss = test_loss / len(test_loader)

    # Compute precision, recall, and F1 for graph and node levels
    graph_precision = precision_score(graph_targets, graph_predictions, average='weighted')
    graph_recall = recall_score(graph_targets, graph_predictions, average='weighted')
    graph_f1 = f1_score(graph_targets, graph_predictions, average='weighted')

    node_precision = precision_score(node_targets, node_predictions, average='weighted')
    node_recall = recall_score(node_targets, node_predictions, average='weighted')
    node_f1 = f1_score(node_targets, node_predictions, average='weighted')

    return graph_accuracy, node_accuracy, avg_test_loss, graph_precision, graph_recall, graph_f1, node_precision, node_recall, node_f1


# Create and write the CSV file headers
with open('BATgraph_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Graph Precision', 'Graph Recall', 'Graph F1'])

with open('BATnode_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Node Precision', 'Node Recall', 'Node F1'])

# Training and testing loop
for epoch in range(100):
    train_loss = train(model, train_loader, optimizer, criterion)
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")
    graph_acc, node_acc, test_loss, graph_precision, graph_recall, graph_f1, node_precision, node_recall, node_f1 = test(
        model, test_loader, criterion)
    print(f"Graph Test Accuracy: {graph_acc:.4f}, Node Test Accuracy: {node_acc:.4f}, Test Loss: {test_loss:.4f}")
    print(
        f"Graph Test Precision: {graph_precision:.4f}, Graph Test Recall: {graph_recall:.4f}, Graph Test F1: {graph_f1:.4f}")
    print(
        f"Node Test Precision: {node_precision:.4f}, Node Test Recall: {node_recall:.4f}, Node Test F1: {node_f1:.4f}")

    # Append the graph results to the graph CSV file
    with open('BATgraph_results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, graph_precision, graph_recall, graph_f1])

    # Append the node results to the node CSV file
    with open('BATnode_results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, node_precision, node_recall, node_f1])

# Final test after all epochs
graph_acc, node_acc, test_loss, graph_precision, graph_recall, graph_f1, node_precision, node_recall, node_f1 = test(
    model, test_loader, criterion)
print(f"Final Graph Test Accuracy: {graph_acc:.4f}, Final Node Test Accuracy: {node_acc:.4f}")
print(f"Final Test Loss: {test_loss:.4f}")
print(
    f"Final Graph Test Precision: {graph_precision:.4f}, Final Graph Test Recall: {graph_recall:.4f}, Final Graph Test F1: {graph_f1:.4f}")
print(
    f"Final Node Test Precision: {node_precision:.4f}, Final Node Test Recall: {node_recall:.4f}, Final Node Test F1: {node_f1:.4f}")
winsound.Beep(1000, 1000)
