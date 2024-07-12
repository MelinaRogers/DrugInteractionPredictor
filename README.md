# Drug Interaction Prediction with GNNS

Predicting drug-drug interactions using Graph Neural Networks and molecular structure data.

##  About

Applying GNN algorithms to predict potential interactions between drugs. I represent the molecular structures as graphs which can capture complex relations and patterns that 
traditional approached may miss. 

## Key Features

- Data loading and preprocessing pipeline for drug interaction datasets
- Conversion of SMILES strings to molecular graphs
- Custom PyTorch Geometric Dataset for drug interactions
- Graph Neural Network model for interaction prediction
- Efficient training and evaluation scripts

## References

This work-in-progress project draws inspiration from (and eventually will expand upon) the following research:

1. [DDI-SMILE: Predicting Drug-Drug Interaction via Smiles String Based Molecule Embedding](https://www.tanvirfarhan.com/publication/ddi_smile/DDI_Smile.pdf)
   by Farhan et al. (2023)

2. [Molecular Graph Enhanced Transformer for Retrosynthesis Prediction](https://arxiv.org/pdf/2006.14002)
   by Mao et al. (2020)

3. [Predicting Drug-Drug Interactions Using Deep Learning Approaches](https://link.springer.com/chapter/10.1007/978-3-031-34107-6_11)
   by Lan et al. (2023)

## Getting Started

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/drug-interaction-gnn.git
cd drug-interaction-gnn
pip install -r requirements.txt