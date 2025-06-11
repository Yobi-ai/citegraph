import torch

from src.models.model1.model import GCN


def test_model_initialization():
    """Test model initialization with different input dimensions."""
    input_dim = 100
    hidden_dim = 64
    output_dim = 10

    model = GCN(input_dim, hidden_dim, output_dim)

    assert model.conv1.in_channels == input_dim
    assert model.conv1.out_channels == hidden_dim
    assert model.linear.out_features == output_dim
    assert isinstance(model, torch.nn.Module)


def test_model_forward():
    """Test model forward pass with dummy data."""
    input_dim = 100
    hidden_dim = 64
    output_dim = 10
    batch_size = 2

    model = GCN(input_dim, hidden_dim, output_dim)

    # Create dummy input
    x = torch.randn(batch_size, input_dim)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    # Forward pass
    output = model(x, edge_index)

    assert output.shape == (batch_size, output_dim)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
