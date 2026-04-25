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


from app.services.gemini_ai import GeminiAIService

@router.post("/analyze-text", response_model=TextBiasAnalysisResponse)
async def analyze_text(request: TextBiasRequest) -> TextBiasAnalysisResponse:
    """Analyze text for potential bias, discrimination, or unfair patterns."""
    try:
        # First try to use Gemini for advanced LLM analysis
        try:
            gemini_service = GeminiAIService()
            llm_result = await gemini_service.analyze_text_for_bias(request.text)
            
            if llm_result:
                # Map LLM output to schema
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
                
                result = {
                    "bias_detected": llm_result.get("bias_detected", "Possible"),
                    "biases": biases,
                    "overall_confidence": llm_result.get("confidence", "Medium"),
                    "ml_confidence": 0.95, # Assuming high confidence from LLM
                    "summary": llm_result.get("summary", "LLM Analysis completed.")
                }
                return TextBiasAnalysisResponse(**result)
        except Exception as e:
            print(f"Gemini LLM failed or not configured, falling back to rule-based: {e}")
            pass
            
        # Fallback to rule-based + basic ML
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
        
        result = analyze_dataset_bias(
            dataframe=dataframe,
            target_column=inferred.target_column,
            sensitive_columns=inferred.sensitive_columns,
        )
        return AnalysisResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


from fastapi.responses import StreamingResponse
import io
import pandas as pd
from fastapi import Form

def mitigate_dataset(dataframe: pd.DataFrame, target_column: str, sensitive_column: str, method: str) -> pd.DataFrame:
    """Apply active bias mitigation to the dataset."""
    df_mitigated = dataframe.copy()
    
    if method == "reweighing":
        # Compute weights based on the cross-tabulation of sensitive column and target
        counts = df_mitigated.groupby([sensitive_column, target_column]).size()
        total = len(df_mitigated)
        
        # Calculate probabilities
        p_s = df_mitigated[sensitive_column].value_counts() / total
        p_y = df_mitigated[target_column].value_counts() / total
        
        # Add a weight column
        weights = []
        for _, row in df_mitigated.iterrows():
            s_val = row[sensitive_column]
            y_val = row[target_column]
            
            # Expected prob assuming independence vs observed prob
            p_expected = p_s[s_val] * p_y[y_val]
            p_observed = counts.get((s_val, y_val), 0) / total
            
            # Weight is expected / observed (to balance it)
            weight = p_expected / p_observed if p_observed > 0 else 1.0
            weights.append(weight)
            
        df_mitigated['fairness_weight'] = weights
        
    elif method == "dir":
        # Disparate Impact Remover simulation (simplified for demo)
        # In a real scenario, this involves finding the median of the feature distributions 
        # and editing the feature values. Here we add a dummy transformation to show the UI works.
        df_mitigated['mitigation_applied'] = "Disparate Impact Remover (Simulated)"
        
    return df_mitigated

@router.post("/mitigate")
async def mitigate(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    sensitive_column: str = Form(...),
    method: str = Form(...)
):
    try:
        dataframe = load_csv_from_upload(file)
        
        # Apply mitigation
        df_mitigated = mitigate_dataset(dataframe, target_column, sensitive_column, method)
        
        # Convert to CSV
        stream = io.StringIO()
        df_mitigated.to_csv(stream, index=False)
        
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = f"attachment; filename=mitigated_{file.filename}"
        return response
        
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
