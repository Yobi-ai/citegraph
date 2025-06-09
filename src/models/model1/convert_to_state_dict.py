import torch
from model import GCN, Model

model = Model(
    r"D:/Sujays documents & files/MS/IDP/Uni Acceptance Letters/DePaul/Classes/Quarter 6/SE489_MLOps/Project/citegraph/models/model_5000.pth"
)

print(model)

torch.save(
    model.state_dict(),
    r"D:/Sujays documents & files/MS/IDP/Uni Acceptance Letters/DePaul/Classes/Quarter 6/SE489_MLOps/Project/citegraph/models/model_5000_state_dict.pth",
)

model2 = GCN(1433, 717, 7)

model2.load_state_dict(
    torch.load(
        r"D:/Sujays documents & files/MS/IDP/Uni Acceptance Letters/DePaul/Classes/Quarter 6/SE489_MLOps/Project/citegraph/models/model_5000_state_dict.pth"
    )
)

print(model2)
