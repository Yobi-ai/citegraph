import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, int(hidden_channels/2))
        self.linear = nn.Linear(int(hidden_channels/2), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)
    
class Model:
    def __init__(self, model_path:str):
        self._model = None
        self.__load(model_path)

    def __load(self, model_path:str):
        self._model = torch.load(model_path, weights_only=False)

    def train(self):
        if self._model is not None:
            self._model.train()
    
    def eval(self):
        if self._model is not None:
            self._model.eval()

    def __call__(self, input:torch.Tensor, edge_index: torch.Tensor):
        if self._model is not None:
            return self._model(input, edge_index)
        
    def __str__(self):
        return f"Model object: \n {self._model}"