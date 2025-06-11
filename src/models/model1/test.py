import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from dataloader import Dataset
from model import GCN, Model
from omegaconf import DictConfig
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)


def test(model, data):
    print("Running Test on Test Set")
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        test_loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask]).item()
        pred = out.argmax(dim=1)
        test_correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        test_acc = test_correct / data.test_mask.sum().item()
        report = classification_report(data.y[data.test_mask], pred[data.test_mask])
        with open("evaluation_report.txt", "w") as outfile:
            outfile.writelines(
                [
                    f"Model Results \n  - Test Loss: {test_loss} \n  - Test Accuracy: {test_acc}\n",
                    f"Classification Report: \n{report}",
                ]
            )
        confmat = confusion_matrix(data.y[data.test_mask], pred[data.test_mask])
        disp = ConfusionMatrixDisplay(confusion_matrix=confmat)
        plt.savefig("confusion_matrix.png")
        disp.plot()
    return test_loss, test_acc


@hydra.main(version_base=None, config_path="confs/test", config_name="inference_conf")
def main(cfg: DictConfig) -> None:
    model = None
    try:
        model = Model(f"{cfg.model_path}/model.pth")
    except Exception:
        model = GCN(1433, 717, 7)
        model.load_state_dict(
            torch.load(f"{cfg.model_state_path}/model_state_dict.pth")
        )

    data = Dataset().load_cora(cfg.data_path)[0]
    test(model, data)
