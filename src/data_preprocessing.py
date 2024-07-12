import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data, Dataset
import logging
import os

class DrugGraphDataset(Dataset):
    """
    A PyTorch Geometric Dataset for drug interaction graphs

    This dataset combines two drug molecules into a single graph for each interaction.
    """

    def __init__(self, root, data, transform=None, pre_transform=None):
        self.data = data
        self.processed_dir = root
        super(DrugGraphDataset, self).__init__(root, transform, pre_transform)

    def process(self):
        """
        Process the raw data into PyTorch Geometric graph objects
        """
        for idx, row in self.data.iterrows():
            mol1 = Chem.MolFromSmiles(row['smiles1'])
            mol2 = Chem.MolFromSmiles(row['smiles2'])
            data1 = mol_to_graph_data(mol1)
            data2 = mol_to_graph_data(mol2)
            combined_data = combine_graphs(data1, data2)
            combined_data.y = torch.tensor([row['interaction_type']], dtype=torch.long)
            torch.save(combined_data, os.path.join(self.processed_dir, f'data_{idx}.pt'))

    def len(self):
        return len(self.data)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))

def mol_to_graph_data(mol):
    """
    Convert an RDKit molecule to a PyTorch Geometric graph

    Args:
        mol: An RDKit molecule object

    Returns:
        A PyTorch Geometric Data object representing the molecule graph
    """
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    
    num_atoms = len(atoms)
    num_bond_features = 3  # bond type, bond stereo, is_conjugated

    x = torch.zeros((num_atoms, 11), dtype=torch.float)
    edge_index = torch.zeros((2, len(bonds) * 2), dtype=torch.long)
    edge_attr = torch.zeros((len(bonds) * 2, num_bond_features), dtype=torch.float)

    for i, atom in enumerate(atoms):
        x[i] = torch.tensor([
            atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(), atom.GetHybridization(),
            atom.GetIsAromatic(), atom.GetMass(), atom.GetExplicitValence(),
            atom.GetImplicitValence(), atom.GetNumExplicitHs(), atom.GetNumImplicitHs()
        ], dtype=torch.float)

    for j, bond in enumerate(bonds):
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index[:, j*2] = torch.tensor([start, end])
        edge_index[:, j*2+1] = torch.tensor([end, start])
        
        edge_attr[j*2] = edge_attr[j*2+1] = torch.tensor([
            bond.GetBondTypeAsDouble(),
            bond.GetStereo(),
            bond.GetIsConjugated()
        ], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def combine_graphs(data1, data2):
    """
    Combine two molecule graphs into a single graph

    This function is used to create a single graph representing a drug interaction.

    Args:
        data1: PyTorch Geometric Data object for the first molecule
        data2: PyTorch Geometric Data object for the second molecule

    Returns:
        A combined PyTorch Geometric Data object
    """
    offset = data1.x.size(0)
    
    x = torch.cat([data1.x, data2.x], dim=0)
    edge_index = torch.cat([data1.edge_index, data2.edge_index + offset], dim=1)
    edge_attr = torch.cat([data1.edge_attr, data2.edge_attr], dim=0)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class DrugDataPreprocessor:
    """
    Preprocessor for drug interaction data

    This class handles loading, preprocessing, and splitting of drug interaction data.
    """

    def __init__(self, input_file: str, output_dir: str):
        self.input_file = input_file
        self.output_dir = output_dir
        self.data = None
        self.dataset = None
        
        # Set up logging - always good to keep track of what's going on
        logging.basicConfig(filename='drug_data_preprocessor.log', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def load_data(self):
        """
        Load the raw data from CSV file
        """
        try:
            self.data = pd.read_csv(self.input_file)
            logging.info(f"Data loaded successfully from {self.input_file}")
        except Exception as e:
            logging.error(f"Error loading data from {self.input_file}: {str(e)}")
            raise

    def prepare_labels(self):
        """
        Prepare labels by encoding interaction types
        """
        try:
            le = LabelEncoder()
            self.data['interaction_type'] = le.fit_transform(self.data['interaction_type'])
            self.label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            logging.info("Labels prepared successfully")
            logging.info(f"Label mapping: {self.label_mapping}")
        except Exception as e:
            logging.error(f"Error preparing labels: {str(e)}")
            raise

    def create_dataset(self):
        """
        Create a DrugGraphDataset from the loaded data
        """
        if self.data is None:
            self.load_data()
        self.prepare_labels()
        
        try:
            self.dataset = DrugGraphDataset(root=self.output_dir, data=self.data)
            logging.info("Graph dataset created successfully")
        except Exception as e:
            logging.error(f"Error creating graph dataset: {str(e)}")
            raise

    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the dataset into training and test sets

        Args:
            test_size: Proportion of the dataset to include in the test split
            random_state: Controls the shuffling applied to the data before applying the split

        Returns:
            A tuple of (train_indices, test_indices)
        """
        if self.dataset is None:
            self.create_dataset()

        try:
            num_samples = len(self.dataset)
            indices = list(range(num_samples))
            train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state, stratify=self.data['interaction_type'])
            
            logging.info(f"Data split into train ({len(train_indices)} samples) and test ({len(test_indices)} samples) sets")
            return train_indices, test_indices
        except Exception as e:
            logging.error(f"Error splitting data: {str(e)}")
            raise

    def preprocess(self):
        """
        Run the entire preprocessing pipeline
        """
        self.create_dataset()
        return self.split_data()

# Example usage
if __name__ == "__main__":
    # TODO: move these paths to a config file
    preprocessor = DrugDataPreprocessor("data/processed/drug_interactions.csv", "data/processed/graph_data")
    train_indices, test_indices = preprocessor.preprocess()
    print(f"Preprocessing complete. Dataset saved in {preprocessor.output_dir}")
    print(f"Number of training samples: {len(train_indices)}")
    print(f"Number of test samples: {len(test_indices)}")
    print("Start trainign")