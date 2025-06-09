import logging
import os
from typing import Dict

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from rich import print

from .dataloader import Dataset
from .model import GCN, Model
from .process_new_files import convert_pdf_to_word_vector

logger = logging.getLogger(__name__)
log_dir = os.path.join(os.path.normpath(os.getcwd()), "logs")
if "logs" not in os.listdir():
    os.mkdir("logs")
FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
# logging.basicConfig(filename=f"{log_dir}/citegraph.log", format=FORMAT, level=logging.INFO)
formatter = logging.Formatter(FORMAT)
file_handler = logging.FileHandler(f"{log_dir}/citegraph_inf.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.propagate = False


class Inference:
    def __init__(self, cfg: DictConfig) -> None:
        try:
            self._model = Model(cfg.model_path)
        except Exception:
            self._model = GCN(1433, 717, 7)
            self._model.load_state_dict(torch.load(cfg.model_state_path))
        self._data = Dataset().load_cora(cfg.data_path)[0]
        self._vocab_path = cfg.word_dict_path
        self._k = cfg.k

        self._label_dict: Dict[int, str] = {
            0: "Theory",
            1: "Reinforcement_Learning",
            2: "Genetic_Algorithms",
            3: "Neural_Networks",
            4: "Probabilistic_Methods",
            5: "Case_Based",
            6: "Rule_Learning",
        }

    def run_sample(self, pdf_path: str) -> str:
        logger.info("Starting Processing File")
        orig_data = self._data
        # orig_edge_index = self._data.edge_index
        word_vector = torch.tensor(
            convert_pdf_to_word_vector(pdf_path, self._vocab_path), dtype=torch.float
        ).unsqueeze(dim=0)
        # print(orig_data.x.shape, word_vector.shape)
        orig_data.x = torch.cat([orig_data.x, word_vector], dim=0)
        new_node_index = orig_data.num_nodes - 1
        # print(new_node_index)

        logger.info("Starting Prediction")
        similarities = F.cosine_similarity(word_vector, orig_data.x)
        top_k_indices = similarities.topk(self._k).indices.tolist()

        edges = torch.tensor(
            [[new_node_index] * len(top_k_indices), top_k_indices], dtype=torch.long
        )
        edges = torch.cat([edges, edges.flip(0)], dim=1)

        orig_data.edge_index = torch.cat([orig_data.edge_index, edges], dim=1)

        self._model.eval()
        with torch.no_grad():
            out = self._model(orig_data.x, orig_data.edge_index)
            pred = out.argmax(dim=1)

        pred_idx = pred[-1].item()
        pred_label = self._label_dict[pred[-1].item()]

        logger.info(f"Predicted: {pred_label}")
        print(f"Prediction:- \n- [bold green]{pred_idx}: {pred_label}[/bold green]")

        print(f"Predicted node connections/similar indexes: {edges}")

        return pred_label


@hydra.main(
    version_base=None, config_path="confs/inference", config_name="inference_conf"
)
def main(cfg: DictConfig) -> None:
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    inf_obj = Inference(cfg)

    # vocab_root_folder = r"D:/Sujays documents & files/MS/IDP/Uni Acceptance Letters/DePaul/Classes/Quarter 6/SE489_MLOps/Project/citegraph/src/data/Cora/CoRA_Raw/"

    pdf_root_path = "pdfs/"
    if "pdfs" not in os.listdir():
        os.mkdir("pdfs")

    pdf_filename = r"Citation Network.pdf"

    print(inf_obj.run_sample(pdf_root_path + pdf_filename))


if __name__ == "__main__":
    logger.info("Started Script")
    main()
