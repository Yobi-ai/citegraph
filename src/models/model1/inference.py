import logging
import os

import hydra
import torch
from dataloader import Dataset
from model import GCN, Model
from omegaconf import DictConfig, OmegaConf
from rich import print
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import mlflow

logger = logging.getLogger(__name__)
log_dir = os.path.join(os.path.normpath(os.getcwd()), 'logs')
if "logs" not in os.listdir():
    os.mkdir("logs")
FORMAT = '%(asctime)s | %(levelname)s | %(message)s'
#logging.basicConfig(filename=f"{log_dir}/citegraph.log", format=FORMAT, level=logging.INFO)
formatter = logging.Formatter(FORMAT)
file_handler = logging.FileHandler(f"{log_dir}/citegraph_inf.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.propagate = False

class Inference:
    def __init__(self, cfg):
        self._model = Model(cfg.model_path)
        self._data = Dataset().load_cora(cfg.data_path)[0]
        self._label_dict = {
            0: "Theory",
            1: "Reinforcement_Learning",
            2: "Genetic_Algorithms",
            3: "Neural_Networks",
            4: "Probabilistic_Methods",
            5: "Case_Based",
            6: "Rule_Learning"
        }
    
    def run_sample(self, idx: int):
        logger.info("Starting Prediction")
        self._model.eval()
        with torch.no_grad():
            out = self._model(self._data.x, self._data.edge_index)
            pred = out.argmax(dim=1)

        pred_idx = pred[idx].item()
        pred_label = self._label_dict[pred[idx].item()]
        
        logger.info(f"Predicted: {pred_label}")
        print(f"Prediction:- \n- [bold green]{pred_idx}: {pred_label}[/bold green]")

        return pred_idx, pred_label

@hydra.main(version_base=None, config_path="confs/inference", config_name="inference_conf")
def main(cfg):
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    inf_obj = Inference(cfg)
    print(inf_obj.run_sample(999))

if __name__ == "__main__":
    logger.info("Started Script")
    main()

