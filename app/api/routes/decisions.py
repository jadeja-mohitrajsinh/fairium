"""
Decision fairness analysis endpoint.

Accepts a CSV containing model predictions + actual outcomes + sensitive attributes,
and returns a full decision-level fairness report.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from typing import Optional
from app.services.inference.data_loader import load_csv_from_upload
from app.services.bias.decision_fairness import analyze_model_decisions, detect_prediction_columns
from app.core.logging import logger

router = APIRouter()


@router.post("/analyze-decisions")
async def analyze_decisions(
    file: UploadFile = File(...),
    prediction_column: Optional[str] = Form(default=None),
    actual_column: Optional[str] = Form(default=None),
    sensitive_columns: Optional[str] = Form(default=None),
):
    """
    Analyze AI model decisions for bias and fairness.

    Upload a CSV with columns:
    - prediction/score column (model output)
    - actual/ground truth column (real outcome)
    - sensitive attribute columns (gender, race, age, etc.)

    Returns decision-level fairness metrics:
    Demographic Parity, Equalized Odds, Equal Opportunity,
    Predictive Parity, FNR Parity, and per-group confusion matrices.
    """
    try:
        df = load_csv_from_upload(file)
        logger.info(f"Decision analysis request: {len(df)} rows, columns: {list(df.columns)}")

        # Auto-detect columns if not provided
        detected = detect_prediction_columns(df)

        pred_col = prediction_column or detected.get("prediction_column")
        actual_col = actual_column or detected.get("actual_column")

        if sensitive_columns:
            sens_cols = [c.strip() for c in sensitive_columns.split(",") if c.strip()]
        else:
            sens_cols = detected.get("sensitive_columns", [])

        # Validate
        if not pred_col:
            raise ValueError(
                "Could not detect a prediction column. "
                "Add a column named 'prediction', 'predicted', 'score', or 'y_pred'."
            )
        if not actual_col:
            raise ValueError(
                "Could not detect an actual outcome column. "
                "Add a column named 'actual', 'true', 'ground_truth', or 'y_true'."
            )
        if pred_col not in df.columns:
            raise ValueError(f"Prediction column '{pred_col}' not found in dataset.")
        if actual_col not in df.columns:
            raise ValueError(f"Actual column '{actual_col}' not found in dataset.")
        if not sens_cols:
            raise ValueError(
                "Could not detect sensitive attribute columns. "
                "Add columns like 'gender', 'race', 'age_group', 'education', etc."
            )

        missing_sens = [c for c in sens_cols if c not in df.columns]
        if missing_sens:
            raise ValueError(f"Sensitive columns not found: {', '.join(missing_sens)}")

        result = analyze_model_decisions(
            df=df,
            prediction_col=pred_col,
            actual_col=actual_col,
            sensitive_columns=sens_cols,
        )

        # Include detected column names in response
        result["detected_columns"] = {
            "prediction_column": pred_col,
            "actual_column": actual_col,
            "sensitive_columns": sens_cols,
        }

        return result

    except ValueError as exc:
        logger.error(f"Validation error in analyze_decisions: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Unexpected error in analyze_decisions: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error during decision analysis")


@router.post("/detect-columns")
async def detect_columns(file: UploadFile = File(...)):
    """
    Preview column detection for a predictions CSV.
    Returns which columns were auto-detected as prediction, actual, and sensitive.
    """
    try:
        df = load_csv_from_upload(file)
        detected = detect_prediction_columns(df)
        return {
            "columns": list(df.columns),
            "detected": detected,
            "row_count": len(df),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
