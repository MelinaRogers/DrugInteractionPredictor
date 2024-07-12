import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from gnn_model import DrugInteractionGNN
from data_loader import DrugDataLoader
from data_preprocessing import mol_to_graph_data, combine_graphs
from rdkit import Chem
import os

def train(model, train_loader, optimizer, device):
    """
    Train the model for one epoch

    Args:
        model: The GNN model
        train_loader: DataLoader with training data
        optimizer: The optimizer for training
        device: Device to train on (CPU or GPU)

    Returns:
        float: Average loss for this epoch
    """
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def test(model, loader, device):
    """
    Test the model on a dataset

    Args:
        model: The GNN model
        loader: DataLoader with test data
        device: Device to test on (CPU or GPU)

    Returns:
        float: Accuracy of the model on the dataset
    """
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

def create_pyg_data(row):
    """
    Create a PyTorch Geometric Data object from a row of the dataset

    Args:
        row: A row from the pandas DataFrame

    Returns:
        Data: A PyTorch Geometric Data object, or None if creation fails
    """
    try:
        mol1 = Chem.MolFromSmiles(row['smiles1'])
        mol2 = Chem.MolFromSmiles(row['smiles2'])
        
        if mol1 is None or mol2 is None:
            print(f"Could not create molecule from SMILES: {row['smiles1']} or {row['smiles2']}")
            return None

        data1 = mol_to_graph_data(mol1)
        data2 = mol_to_graph_data(mol2)
        combined_data = combine_graphs(data1, data2)
        
        combined_data.y = torch.tensor([row['interaction_type']], dtype=torch.long)
        
        return combined_data
    except Exception as e:
        print(f"Error processing row: {e}")
        return None

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare dataset
    # TODO: Move this path to a config file
    loader = DrugDataLoader("path/to/ddi_corpus")
    dataset = loader.create_dataset()
    print(f"Dataset created with {len(dataset)} entries")
    print(f"Columns: {dataset.columns}")
    print(dataset.head())  # Let's see what we're working with

    if len(dataset) == 0:
        print("Dataset is empty. Check DrugDataLoader ")
        return

    # Split dataset
    # Note: I might want to use stratified sampling here in the future
    train_dataset = dataset.sample(frac=0.8, random_state=42)
    test_dataset = dataset.drop(train_dataset.index)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Convert pandas DataFrames to PyTorch Geometric Data objects
    train_data_list = [create_pyg_data(row) for _, row in train_dataset.iterrows()]
    train_data_list = [data for data in train_data_list if data is not None]
    test_data_list = [create_pyg_data(row) for _, row in test_dataset.iterrows()]
    test_data_list = [data for data in test_data_list if data is not None]
    
    print(f"Processed train data size: {len(train_data_list)}")
    print(f"Processed test data size: {len(test_data_list)}")

    if len(train_data_list) == 0 or len(test_data_list) == 0:
        print("pyg_data debugging needed :(")
        return
    
    # Create data loaders
    # might adjust batch size based on available memory
    train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data_list, batch_size=32)

    # Initialize model
    model = DrugInteractionGNN(
        num_node_features=train_data_list[0].x.size(1),
        num_edge_features=train_data_list[0].edge_attr.size(1),
        hidden_channels=64,  # read more, and determine best way to experiment with this 
        num_classes=len(loader.label_mapping)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(1, 201):  # 200 epochs 
        loss = train(model, train_loader, optimizer, device)
        train_acc = test(model, train_loader, device)
        test_acc = test(model, test_loader, device)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    # Save the model and label mapping
    torch.save(model.state_dict(), 'drug_interaction_gnn_model.pth')
    print(f"Model saved to {os.path.abspath('drug_interaction_gnn_model.pth')}")

    torch.save(loader.label_mapping, 'label_mapping.pth')
    print(f"Label mapping saved to {os.path.abspath('label_mapping.pth')}")

if __name__ == '__main__':
    main()