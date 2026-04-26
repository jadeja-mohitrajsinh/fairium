from fastapi import APIRouter, File, UploadFile, HTTPException
from app.schemas.analysis import AnalysisResponse, TextBiasAnalysisResponse, TextBiasRequest
from app.services.inference.data_loader import load_csv_from_upload
from app.services.inference.engine import infer_analysis_columns
from app.services.inference.validator import validate_analysis_input
from app.services.bias.fairness import analyze_dataset_bias
from app.services.ai.text_bias import TextBiasAnalyzer
from app.services.ai.gemini import GeminiAIService
from app.core.logging import logger

router = APIRouter()

@router.post("/analyze-text", response_model=TextBiasAnalysisResponse)
async def analyze_text(request: TextBiasRequest) -> TextBiasAnalysisResponse:
    """Analyze text for potential bias, discrimination, or unfair patterns."""
    try:
        # Try Gemini first
        try:
            gemini_service = GeminiAIService()
            llm_result = await gemini_service.analyze_text_for_bias(request.text)
            if llm_result:
                biases = []
                for b in llm_result.get("biases", []):
                    biases.append({
                        "type": b.get("type", "Unknown"),
                        "confidence": llm_result.get("confidence", "Medium"),
                        "explanation": b.get("explanation", ""),
                        "alternatives": b.get("alternatives", []),
                        "keyword_matches": [],
                        "phrase_matches": [],
                        "ambiguous_matches": []
                    })
                
                return TextBiasAnalysisResponse(
                    bias_detected=llm_result.get("bias_detected", "Possible"),
                    biases=biases,
                    overall_confidence=llm_result.get("confidence", "Medium"),
                    ml_confidence=0.95,
                    summary=llm_result.get("summary", "LLM Analysis completed.")
                )
        except Exception as e:
            logger.warning(f"Gemini LLM analysis failed, falling back to rule-based: {e}")
            
        # Fallback to rule-based + ML
        result = TextBiasAnalyzer.analyze_text(request.text)
        return TextBiasAnalysisResponse(**result)
    except Exception as exc:
        logger.error(f"Error in analyze_text: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error during text analysis")

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)) -> AnalysisResponse:
    """Analyze a dataset for potential bias across multiple attributes."""
    try:
        dataframe = load_csv_from_upload(file)
        
        # Auto-detect target and sensitive columns
        inferred = infer_analysis_columns(dataframe)
        
        # Validate input (converting sensitive_columns list to comma-separated string for validator if needed, 
        # but better to update validator or handle here)
        validated = validate_analysis_input(
            dataframe=dataframe,
            target_column=inferred.target_column,
            sensitive_columns=",".join(inferred.sensitive_columns)
        )
        
        result = analyze_dataset_bias(
            dataframe=validated.dataframe,
            target_column=validated.target_column,
            sensitive_columns=validated.sensitive_columns,
        )
        return AnalysisResponse(**result)
    except ValueError as exc:
        logger.error(f"Validation error in analyze: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Unexpected error in analyze: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error during dataset analysis")
