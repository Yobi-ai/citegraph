import os
from typing import Dict

from fastapi import APIRouter, File, UploadFile
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


@router.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Inferences on an uploaded PDF file.

    Args:
        file: The PDF file to analyze

    Returns:
        Dictionary containing the predicted label
    """
    # Save uploaded file temporarily
    temp_file_path = f"/tmp/{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    try:
        inference = Inference(cfg)
        predicted_label = inference.run_sample(temp_file_path)

        # Clean up temporary file
        os.remove(temp_file_path)

        return {"predicted_label": predicted_label}
    except Exception as e:
        # Clean up temporary file in case of error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise e
