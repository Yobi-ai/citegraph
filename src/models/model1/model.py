import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print
from torch_geometric.nn import GCNConv

logger = logging.getLogger(__name__)
log_dir = os.path.join(os.path.normpath(os.getcwd()), 'logs')
if "logs" not in os.listdir():
    os.mkdir("logs")
FORMAT = '%(asctime)s | %(levelname)s | %(message)s'
#logging.basicConfig(filename=f"{log_dir}/citegraph.log", format=FORMAT, level=logging.INFO)
formatter = logging.Formatter(FORMAT)
file_handler = logging.FileHandler(f"{log_dir}/citegraph.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.propagate = False

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
        logger.info('Model Loaded')
        print("[bold green]Model Loaded Successfully[/bold green]")

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