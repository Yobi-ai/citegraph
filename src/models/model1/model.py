from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn
import torch

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