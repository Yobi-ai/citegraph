import torch
from dataloader import Dataset
from model import GCN, Model
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


class Inference:
    def __init__(self, model_path: str, data_path: str):
        self._model = Model(model_path)
        self._data = Dataset().load_cora(data_path)
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
        self._model.eval()
        with torch.no_grad():
            out = self._model(self._data.x, self._data.edge_index)
            #print(out)
            #test_loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask]).item()
            pred = out.argmax(dim=1)
            #test_indices = data.test_mask.nonzero(as_tuple=True)[0]
            #print(pred)
            #test_correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
            #test_acc = test_correct / data.test_mask.sum().item()
        return self._label_dict[pred[idx].item()]

if __name__ == "__main__":
    #model_path = "D:/Sujays documents & files/MS/IDP/Uni Acceptance Letters/DePaul/Classes/Quarter 6/SE489_MLOps/Project/citegraph/models/model_5000.pth"
    model_path = "../../../models/model_5000.pth"
    
    data_path = "../../data/"

    inf_obj = Inference(model_path, data_path)

    print(inf_obj.run_sample(999))