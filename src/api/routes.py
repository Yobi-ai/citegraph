import os
from typing import Dict

from fastapi import APIRouter
from omegaconf import OmegaConf

from ..models.model1.inference import Inference

router = APIRouter()

# Load config
config_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "model1",
    "confs",
    "inference",
    "inference_conf.yaml",
)
cfg = OmegaConf.load(config_path)


@router.get("/")
async def root() -> Dict[str, str]:
    """
    Root endpoint that returns a simple greeting.
    """
    return {"message": "Welcome to citegraph api"}


@router.get("/api/predict")
async def predict() -> str:
    """
    Inferences on a file.

    Returns:
        Label of the predicted class for the file.
    """
    vocab_root_folder = r"app/src/data/Cora/CoRA_Raw/"
    pdf_root_path = "app/docs/"
    pdf_filename = r"Citation Network.pdf"

    inference = Inference(cfg)
    predicted_label: str = inference.run_sample(
        pdf_root_path + pdf_filename, vocab_root_folder + "final_words_dictionary.txt"
    )

    return predicted_label
