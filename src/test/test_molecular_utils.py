import unittest
from utils.molecular_utils import canonicalize_smiles, calculate_molecular_properties, smiles_to_fingerprint, check_drug_likeness
from rdkit import Chem

class TestMolecularUtils(unittest.TestCase):
    def test_canonicalize_smiles(self):
        smiles = "C(C(=O)O)N"
        canonical_smiles = canonicalize_smiles(smiles)
        self.assertEqual(canonical_smiles, "NCC(=O)O")

    def test_calculate_molecular_properties(self):
        mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
        properties = calculate_molecular_properties(mol)
        self.assertIn('MolWt', properties)
        self.assertIn('LogP', properties)
        self.assertIn('NumHDonors', properties)
        self.assertIn('NumHAcceptors', properties)

    def test_smiles_to_fingerprint(self):
        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        fp = smiles_to_fingerprint(smiles)
        self.assertIsNotNone(fp)
        self.assertEqual(len(fp), 2048)

    def test_check_drug_likeness(self):
        drug_like_mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin
        non_drug_like_mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCCCCCC")  # Long alkane chain

        self.assertTrue(check_drug_likeness(drug_like_mol))
        self.assertFalse(check_drug_likeness(non_drug_like_mol))

if __name__ == '__main__':
    unittest.main()