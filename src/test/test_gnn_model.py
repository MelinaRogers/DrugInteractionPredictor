import unittest
import torch
from src.gnn_model import DrugInteractionGNN

class TestDrugInteractionGNN(unittest.TestCase):
    def setUp(self):
        self.model = DrugInteractionGNN(num_node_features=11, num_edge_features=3, hidden_channels=64, num_classes=2)

    def test_model_output_shape(self):
        # Create dummy input data
        x = torch.randn(10, 11)  # 10 nodes, 11 features each
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        edge_attr = torch.randn(4, 3)  # 4 edges, 3 features each
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)  # Two graphs

        output = self.model(x, edge_index, edge_attr, batch)
        self.assertEqual(output.shape, (2, 2))  # 2 graphs, 2 classes

    def test_model_parameters(self):
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(torch.isnan(param).any())

if __name__ == '__main__':
    unittest.main()