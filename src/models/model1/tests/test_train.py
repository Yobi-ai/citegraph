import pytest
import torch

from ..dataloader import Dataset
from ..model import GCN
from ..train import Trainer


@pytest.fixture
def dummy_data():
    """Create dummy data for testing."""
    x = torch.randn(10, 100)  # 10 nodes, 100 features
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    y = torch.randint(0, 2, (10,))  # Binary labels
    return Dataset(x, edge_index, y)


def test_trainer_initialization(dummy_data):
    """Test trainer initialization."""
    model = GCN(100, 64, 2)  # input_dim=100, hidden_dim=64, output_dim=2
    trainer = Trainer(model, dummy_data, lr=0.01, epochs=2)

    assert trainer.model == model
    assert trainer.data == dummy_data
    assert trainer.optimizer is not None
    assert trainer.criterion is not None


def test_train_epoch(dummy_data):
    """Test a single training epoch."""
    model = GCN(100, 64, 2)
    trainer = Trainer(model, dummy_data, lr=0.01, epochs=1)

    # Run one epoch
    loss = trainer._train_epoch()

    assert isinstance(loss, float)
    assert not torch.isnan(torch.tensor(loss))
    assert not torch.isinf(torch.tensor(loss))
