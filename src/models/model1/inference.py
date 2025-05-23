import logging
import os
import torch
from dataloader import Dataset
from model import GCN, Model
from rich import print
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

logger = logging.getLogger(__name__)
log_dir = os.path.join(os.path.normpath(os.getcwd()), 'logs')
if "logs" not in os.listdir():
    os.mkdir("logs")
FORMAT = '%(asctime)s | %(levelname)s | %(message)s'
logging.basicConfig(filename=f"{log_dir}/citegraph.log", format=FORMAT, level=logging.INFO)

class Inference:
    def __init__(self, model_path: str, data_path: str):
        self._model = Model(model_path)
        self._data = Dataset().load_cora(data_path)[0]
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

        pred_label = self._label_dict[pred[idx].item()]
        
        logger.info(f"Predicted: {pred_label}")
        print(f"[bold green]- Prediction: {pred_label}[/bold green]")

        return pred_label

if __name__ == "__main__":
    logger.info("Started Script")
    #model_path = "D:/Sujays documents & files/MS/IDP/Uni Acceptance Letters/DePaul/Classes/Quarter 6/SE489_MLOps/Project/citegraph/models/model_5000.pth"
    model_path = "../../../models/model_5000.pth"
    
    data_path = "../../data/"

    inf_obj = Inference(model_path, data_path)
    print(inf_obj.run_sample(999))