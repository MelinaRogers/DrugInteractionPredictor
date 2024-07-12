import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch

class DrugInteractionGNN(torch.nn.Module):
    """
    Graph Neural Network for predicting drug interactions

    Args:
        num_node_features (int): Number of features for each node
        num_edge_features (int): Number of features for each edge
        hidden_channels (int): Number of hidden channels in the GNN layers
        num_classes (int): Number of output classes

    TODO: Consider adding skip connections to improve gradient flow
    """
    def __init__(self, num_node_features, num_edge_features, hidden_channels, num_classes):
        super(DrugInteractionGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        # Node embedding
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv3(x, edge_index, edge_attr)
        
        # Readout - might want to experiment with other pooling methods later
        x = global_mean_pool(x, batch)
        
        # Prediction
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=-1)

def train(model, loader, optimizer, device):
    """
    Train the model for one epoch

    Args:
        model (DrugInteractionGNN): The model to train
        loader (DataLoader): DataLoader containing the training data
        optimizer (torch.optim.Optimizer): The optimizer to use
        device (torch.device): The device to train on

    Returns:
        float: Average loss for this epoch
    """
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def test(model, loader, device):
    """
    Test the model on a dataset

    Args:
        model (DrugInteractionGNN): The model to test
        loader (DataLoader): DataLoader containing the test data
        device (torch.device): The device to test on

    Returns:
        float: Accuracy of the model on the test set
    """
    model.eval()
    total_correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        pred = out.argmax(dim=1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)

def predict(model, data, device):
    """
    Make predictions using the trained model

    Args:
        model (DrugInteractionGNN): The trained model
        data (Data): Input data
        device (torch.device): The device to run predictions on

    Returns:
        tuple: Predicted class and class probabilities
    """
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        prob = torch.exp(out)
        pred = out.argmax(dim=1)
    return pred, prob

# Example usage - might need tweaking based on specific setup
if __name__ == "__main__":
    from torch_geometric.loader import DataLoader
    from src.data_preprocessing import DrugDataPreprocessor

    # TODO: Move these paths to a config file
    preprocessor = DrugDataPreprocessor("data/processed/drug_interactions.csv", "data/processed/graph_data")
    dataset = preprocessor.dataset
    train_indices, test_indices = preprocessor.split_data()

    train_dataset = [dataset[i] for i in train_indices]
    test_dataset = [dataset[i] for i in test_indices]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DrugInteractionGNN(num_node_features=11, num_edge_features=3, hidden_channels=64, num_classes=len(preprocessor.label_mapping)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Main training loop
    for epoch in range(1, 201):  # Thinking about implementing early stopping here
        loss = train(model, train_loader, optimizer, device)
        train_acc = test(model, train_loader, device)
        test_acc = test(model, test_loader, device)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    # Save the trained model - remember to also save the label mapping somewhere
    torch.save(model.state_dict(), 'drug_interaction_gnn_model.pth')
    print("Training complete. Model saved to 'drug_interaction_gnn_model.pth'")