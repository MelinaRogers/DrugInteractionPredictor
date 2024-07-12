import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from flask import Flask, render_template, request, jsonify
import torch
from rdkit import Chem
from src.gnn_model import DrugInteractionGNN
from src.data_preprocessing import mol_to_graph_data, combine_graphs
from src.utils.molecular_utils import canonicalize_smiles

app = Flask(__name__)

# Load model
model = DrugInteractionGNN(num_node_features=11, num_edge_features=3, hidden_channels=64, num_classes=3) 
model.load_state_dict(torch.load(os.path.join(project_root, 'drug_interaction_gnn_model.pth')))
model.eval()

# Load maps
label_mapping = torch.load(os.path.join(project_root, 'label_mapping.pth'))
inverse_label_mapping = {v: k for k, v in label_mapping.items()}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    smiles1 = request.form['smiles1']
    smiles2 = request.form['smiles2']

    smiles1 = canonicalize_smiles(smiles1)
    smiles2 = canonicalize_smiles(smiles2)

    if smiles1 is None or smiles2 is None:
        return jsonify({'error': 'Invalid SMILES string(s)'})

    # Convert SMILES to graph data
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return jsonify({'error': 'Could not convert SMILES to molecule'})

    data1 = mol_to_graph_data(mol1)
    data2 = mol_to_graph_data(mol2)
    combined_data = combine_graphs(data1, data2)

    # Make prediction
    with torch.no_grad():
        output = model(combined_data.x, combined_data.edge_index, combined_data.edge_attr, combined_data.batch)
        probabilities = torch.exp(output)
        predicted_class = torch.argmax(probabilities).item()

    interaction = inverse_label_mapping[predicted_class]
    confidence = probabilities[0][predicted_class].item()

    return jsonify({
        'interaction': interaction,
        'confidence': f'{confidence:.2f}'
    })

if __name__ == '__main__':
    app.run(debug=True)