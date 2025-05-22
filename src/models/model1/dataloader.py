import logging
import os

from rich import print
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

logger = logging.getLogger(__name__)
log_dir = os.path.join(os.path.normpath(os.getcwd()), 'logs')
if "logs" not in os.listdir():
    os.mkdir("logs")
FORMAT = '%(asctime)s | %(levelname)s | %(message)s'
logging.basicConfig(filename=f"{log_dir}/citegraph.log", format=FORMAT, level=logging.INFO)

class Dataset:
    def __init__(self):
        self.dataset = None
        #return self.__load_cora(data_path)

    def load_cora(self, data_path):
        self.dataset = Planetoid(root=data_path, name='Cora', transform=NormalizeFeatures())
        logger.info('Dataset Loaded')
        print("[bold green]Dataset Loaded Successfully[/bold green]")
        return self.dataset[0]