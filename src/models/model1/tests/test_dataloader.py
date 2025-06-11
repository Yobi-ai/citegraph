import torch

from ..dataloader import Dataset


def test_dataset_initialization():
    """Test dataset initialization with dummy data."""
    # Create dummy data
    x = torch.randn(10, 100)  # 10 nodes, 100 features
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    y = torch.randint(0, 2, (10,))  # Binary labels

    dataset = Dataset(x, edge_index, y)

    assert dataset.x.shape == (10, 100)
    assert dataset.edge_index.shape == (2, 3)
    assert dataset.y.shape == (10,)
    assert dataset.num_nodes == 10
    assert dataset.num_features == 100


def test_dataset_get_item():
    """Test dataset indexing."""
    x = torch.randn(10, 100)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    y = torch.randint(0, 2, (10,))

    dataset = Dataset(x, edge_index, y)

    # Test getting a single item
    data = dataset[0]

    assert hasattr(data, "x")
    assert hasattr(data, "edge_index")
    assert hasattr(data, "y")
    assert data.x.shape == (10, 100)
    assert data.edge_index.shape == (2, 3)
    assert data.y.shape == (10,)
