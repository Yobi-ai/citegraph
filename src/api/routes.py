from typing import Dict

from fastapi import APIRouter

from models.model1.inference import run_sample

router = APIRouter()


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

    vocab_root_folder = r"D:/Sujays documents & files/MS/IDP/Uni Acceptance Letters/DePaul/Classes/Quarter 6/SE489_MLOps/Project/citegraph/src/data/Cora/CoRA_Raw/"

    pdf_root_path = "pdfs/"

    pdf_filename = r"Citation Network.pdf"

    predicted_label: str = run_sample(
        pdf_root_path + pdf_filename, vocab_root_folder + "final_words_dictionary.txt"
    )

    return predicted_label
