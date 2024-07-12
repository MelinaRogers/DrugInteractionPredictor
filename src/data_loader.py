import requests
import pandas as pd
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict
import logging
import time
from functools import lru_cache
from rapidfuzz import fuzz, process
from multiprocessing import Pool
import json

class DrugDataLoader:
    """
    Loads and processes drug interaction data from the DDI corpus
    """
    def __init__(self, ddi_corpus_path: str, cache_file: str = 'smiles_cache.json'):
        self.ddi_corpus_path = ddi_corpus_path
        self.pubchem_base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        self.cache_file = cache_file
        self.smiles_cache = self.load_cache()
        self.last_request_time = 0
        self.request_interval = 0.2  # 200ms between requests, might need tweaking
        
        # Set up logging - we'll want to keep an eye on this
        logging.basicConfig(filename='drug_data_loader.log', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def load_cache(self) -> Dict[str, str]:
        """Load the SMILES cache from file, if it exists"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def save_cache(self):
        """Save the SMILES cache to file"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.smiles_cache, f)

    @lru_cache(maxsize=1000)
    def get_smiles(self, compound_name: str) -> str:
        """
        Fetch SMILES representation of a compound from PubChem
        
        Uses rate limiting and caching to avoid hammering the PubChem API
        """
        if compound_name in self.smiles_cache:
            return self.smiles_cache[compound_name]

        # Rate limiting - be nice to PubChem!
        current_time = time.time()
        if current_time - self.last_request_time < self.request_interval:
            time.sleep(self.request_interval - (current_time - self.last_request_time))
        self.last_request_time = time.time()

        url = f"{self.pubchem_base_url}/compound/name/{compound_name}/property/IsomericSMILES/TXT"
        try:
            response = requests.get(url)
            response.raise_for_status()
            smiles = response.text.strip()
            self.smiles_cache[compound_name] = smiles
            self.save_cache()
            return smiles
        except requests.RequestException as e:
            logging.error(f"Error fetching SMILES for {compound_name}: {str(e)}")
            return None

    def load_ddi_corpus(self) -> List[Tuple[str, str, str]]:
        """
        Load and parse the DDI Corpus XML files
        
        Returns a list of (drug1, drug2, interaction_type) tuples
        """
        interactions = []
        for root, dirs, files in os.walk(self.ddi_corpus_path):
            for file in files:
                if file.endswith('.xml'):
                    file_path = os.path.join(root, file)
                    try:
                        tree = ET.parse(file_path)
                        root = tree.getroot()
                        for sentence in root.findall('.//sentence'):
                            for pair in sentence.findall('pair'):
                                e1 = pair.get('e1')
                                e2 = pair.get('e2')
                                interaction_type = pair.get('ddi-type')
                                if interaction_type:  # Only consider pairs with interaction
                                    drug1 = sentence.find(f".//entity[@id='{e1}']").get('text')
                                    drug2 = sentence.find(f".//entity[@id='{e2}']").get('text')
                                    interactions.append((drug1, drug2, interaction_type))
                    except ET.ParseError as e:
                        logging.error(f"Error parsing XML file {file_path}: {str(e)}")
        return interactions

    def fuzzy_match_drug(self, drug_name: str) -> str:
        """
        Fuzzy match drug name with PubChem compound names
        
        This is a bit of a hack, might need improvement later
        """
        known_drugs = list(self.smiles_cache.keys())
        best_match = max(known_drugs, key=lambda x: fuzz.ratio(drug_name, x), default=drug_name)
        return best_match if fuzz.ratio(drug_name, best_match) > 80 else drug_name

    def process_interaction(self, interaction: Tuple[str, str, str]) -> Dict:
        """Process a single drug interaction"""
        drug1, drug2, interaction_type = interaction
        drug1 = self.fuzzy_match_drug(drug1)
        drug2 = self.fuzzy_match_drug(drug2)
        smiles1 = self.get_smiles(drug1)
        smiles2 = self.get_smiles(drug2)
        return {
            'drug1': drug1,
            'drug2': drug2,
            'smiles1': smiles1,
            'smiles2': smiles2,
            'interaction_type': interaction_type
        }

    def create_dataset(self) -> pd.DataFrame:
        """
        Create dataset combining drug interactions and SMILES representations
        
        Uses parallel processing to speed things up a bit
        """
        interactions = self.load_ddi_corpus()
        with Pool() as pool:
            data = pool.map(self.process_interaction, interactions)
        return pd.DataFrame([d for d in data if d['smiles1'] and d['smiles2']])

    def save_dataset(self, output_path: str):
        """Save the created dataset to a CSV file"""
        df = self.create_dataset()
        df.to_csv(output_path, index=False)
        logging.info(f"Dataset saved to {output_path}")
        print(f"Dataset saved to {output_path}")  # Let the user know we're done

# Example usage
if __name__ == "__main__":
    # TODO: Move these paths to a config file
    loader = DrugDataLoader("path/to/ddi_corpus", "smiles_cache.json")
    loader.save_dataset("data/processed/drug_interactions.csv")