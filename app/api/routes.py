from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.models.schemas import AnalysisResponse, TextBiasAnalysisResponse
from app.services.data_loader import load_csv_from_upload
from app.services.inference import infer_analysis_columns
from app.services.fairness import analyze_dataset_bias
from app.services.text_bias import TextBiasAnalyzer


router = APIRouter()


class TextBiasRequest(BaseModel):
    text: str


@router.post("/analyze-text", response_model=TextBiasAnalysisResponse)
async def analyze_text(request: TextBiasRequest) -> TextBiasAnalysisResponse:
    """Analyze text for potential bias, discrimination, or unfair patterns."""
    try:
        result = TextBiasAnalyzer.analyze_text(request.text)
        return TextBiasAnalysisResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    file: UploadFile = File(...),
) -> AnalysisResponse:
    try:
        dataframe = load_csv_from_upload(file)
        
        # Auto-detect target and sensitive columns from dataset
        inferred = infer_analysis_columns(dataframe)
        
        result = await analyze_dataset_bias(
            dataframe=dataframe,
            target_column=inferred.target_column,
            sensitive_columns=inferred.sensitive_columns,
        )
        return AnalysisResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
