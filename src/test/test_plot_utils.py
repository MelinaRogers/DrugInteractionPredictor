import unittest
import matplotlib.pyplot as plt
from visualization.plot_utils import plot_molecule, plot_training_history, plot_confusion_matrix
import numpy as np

class TestPlotUtils(unittest.TestCase):
    def test_plot_molecule(self):
        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        plot_molecule(smiles, title="Test Molecule")
        plt.close()

    def test_plot_training_history(self):
        history = {
            'train_loss': [0.5, 0.4, 0.3],
            'val_loss': [0.6, 0.5, 0.4],
            'train_acc': [0.7, 0.8, 0.9],
            'val_acc': [0.6, 0.7, 0.8]
        }
        plot_training_history(history)
        plt.close()

    def test_plot_confusion_matrix(self):
        cm = np.array([[10, 2], [3, 15]])
        class_names = ['Class 0', 'Class 1']
        plot_confusion_matrix(cm, class_names)
        plt.close()

if __name__ == '__main__':
    unittest.main()