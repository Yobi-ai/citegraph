from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


class Dataset:
    def __init__(self):
        self.dataset = None
        #return self.__load_cora(data_path)

    def load_cora(self, data_path):
        self.dataset = Planetoid(root=data_path, name='Cora', transform=NormalizeFeatures())
        return self.dataset[0]