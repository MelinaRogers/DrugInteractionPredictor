# visualization/plot_utils.py

import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw
import networkx as nx
from torch_geometric.utils import to_networkx

def plot_molecule(smiles, title=None):
    """
    Plot a 2D representation of a molecule from its SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()

def plot_training_history(history):
    """
    Plot training and validation loss/accuracy over epochs.
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, class_names):
    """
    Plot a confusion matrix.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def plot_graph(data, title=None):
    """
    Plot a graph representation of a molecule.
    """
    G = to_networkx(data, to_undirected=True)
    
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
    
    edge_labels = {(u, v): '' for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def plot_property_distribution(properties, property_name):
    """
    Plot the distribution of a molecular property.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(properties, kde=True)
    plt.xlabel(property_name)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {property_name}')
    plt.show()