# utils/molecular_utils.py
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import numpy as np

def canonicalize_smiles(smiles):
    """
    Convert SMILES to a canonical form

    Args:
        smiles (str): The input SMILES string

    Returns:
        str or None: The canonicalized SMILES string, or None if the input is invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    return None

def calculate_molecular_properties(mol):
    """
    Calculate basic molecular properties for a given RDKit molecule

    Args:
        mol (rdkit.Chem.rdchem.Mol): An RDKit molecule object

    Returns:
        dict: A dictionary containing various molecular properties:
            - MolWt: Molecular weight
            - LogP: Octanol-water partition coefficient
            - NumHDonors: Number of hydrogen bond donors
            - NumHAcceptors: Number of hydrogen bond acceptors
            - NumRotatableBonds: Number of rotatable bonds
            - NumHeteroatoms: Number of heteroatoms
            - NumRings: Number of rings
            - TPSA: Topological Polar Surface Area
    """
    return {
        'MolWt': Descriptors.ExactMolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
        'NumRings': Descriptors.RingCount(mol),
        'TPSA': Descriptors.TPSA(mol)
    }

def smiles_to_fingerprint(smiles, fp_type='morgan', radius=2, nBits=2048):
    """
    Convert SMILES to molecular fingerprint

    Args:
        smiles (str): The input SMILES string
        fp_type (str): The type of fingerprint to generate. Options are 'morgan' or 'maccs'
        radius (int): The radius for Morgan fingerprints (used only if fp_type is 'morgan')
        nBits (int): The number of bits for Morgan fingerprints (used only if fp_type is 'morgan')

    Returns:
        numpy.ndarray or None: The molecular fingerprint as a numpy array, or None if the input is invalid

    Raises:
        ValueError: If an unsupported fingerprint type is specified
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    if fp_type == 'morgan':
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    elif fp_type == 'maccs':
        fp = AllChem.GetMACCSKeysFingerprint(mol)
    else:
        raise ValueError(f"Unsupported fingerprint type: {fp_type}")
    
    return np.array(fp)

def check_drug_likeness(mol):
    """
    Check if a molecule satisfies Lipinski's Rule of Five

    Args:
        mol (rdkit.Chem.rdchem.Mol): An RDKit molecule object

    Returns:
        bool: True if the molecule satisfies at least 3 of the 4 Lipinski rules, False otherwise
    """
    mol_props = calculate_molecular_properties(mol)
    
    rules = [
        mol_props['MolWt'] <= 500,
        mol_props['LogP'] <= 5,
        mol_props['NumHDonors'] <= 5,
        mol_props['NumHAcceptors'] <= 10
    ]
    
    return sum(rules) >= 3  # a compound is considered drug-like if it satisfies at least 3 of the 4 rules

def get_atom_features(atom):
    """
    Get relevant features for an atom

    Args:
        atom (rdkit.Chem.rdchem.Atom): An RDKit atom object

    Returns:
        list: A list of atomic features including atomic number, degree, formal charge,
              number of radical electrons, hybridization, aromaticity, mass, valence,
              and number of hydrogens
    """
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetNumRadicalElectrons(),
        atom.GetHybridization(),
        atom.GetIsAromatic(),
        atom.GetMass(),
        atom.GetExplicitValence(),
        atom.GetImplicitValence(),
        atom.GetNumExplicitHs(),
        atom.GetNumImplicitHs()
    ]

def get_bond_features(bond):
    """
    Get relevant features for a bond

    Args:
        bond (rdkit.Chem.rdchem.Bond): An RDKit bond object

    Returns:
        list: A list of bond features including bond type, conjugation, and ring membership
    """
    return [
        bond.GetBondTypeAsDouble(),
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]