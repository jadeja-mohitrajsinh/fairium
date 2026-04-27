from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from app.core.logging import logger

router = APIRouter()

SAMPLE_DATASETS = [
    {
        "id": "adult",
        "name": "Adult Income (Census)",
        "description": "Predict whether income exceeds $50K/yr. Classic fairness benchmark with gender & race attributes.",
        "domain": "income",
        "rows": 32561,
        "filename": "adult.csv",
    },
    {
        "id": "hr_attrition",
        "name": "IBM HR Employee Attrition",
        "description": "Predict employee attrition. Contains gender, age, department, and job role attributes.",
        "domain": "hiring",
        "rows": 1470,
        "filename": "WA_Fn-UseC_-HR-Employee-Attrition.csv",
    },
    {
        "id": "german_credit",
        "name": "German Credit Risk",
        "description": "Predict credit risk. Classic lending dataset with age, sex, and financial attributes.",
        "domain": "lending",
        "rows": 1000,
        "filename": "german_credit_data.csv",
    },
    {
        "id": "compas",
        "name": "COMPAS Recidivism",
        "description": "Predict recidivism risk. Widely studied for racial bias in criminal justice.",
        "domain": "criminal_justice",
        "rows": 7214,
        "filename": "compas-scores-raw.csv",
    },
]

DATA_DIR = Path("data/New folder")


@router.get("/datasets")
async def list_datasets():
    """List available sample datasets for demo."""
    available = []
    for ds in SAMPLE_DATASETS:
        path = DATA_DIR / ds["filename"]
        if path.exists():
            available.append({k: v for k, v in ds.items() if k != "filename"})
    return {"datasets": available}


@router.get("/datasets/{dataset_id}/download")
async def download_dataset(dataset_id: str):
    """Download a sample dataset by ID."""
    dataset = next((d for d in SAMPLE_DATASETS if d["id"] == dataset_id), None)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    path = DATA_DIR / dataset["filename"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found on server.")

    logger.info(f"Serving sample dataset: {dataset_id}")
    return FileResponse(
        path=str(path),
        media_type="text/csv",
        filename=dataset["filename"],
    )
