from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
import io
import pandas as pd
from app.services.inference.data_loader import load_csv_from_upload
from app.services.mitigation.strategies import apply_active_mitigation
from app.core.logging import logger

router = APIRouter()

@router.post("/mitigate")
async def mitigate(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    sensitive_column: str = Form(...),
    method: str = Form(...)
):
    """Apply bias mitigation to a dataset and return the modified file."""
    try:
        dataframe = load_csv_from_upload(file)
        
        # Apply mitigation via service
        df_mitigated = apply_active_mitigation(dataframe, target_column, sensitive_column, method)
        
        # Convert to CSV for streaming
        stream = io.StringIO()
        df_mitigated.to_csv(stream, index=False)
        
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = f"attachment; filename=mitigated_{file.filename}"
        return response
        
    except ValueError as exc:
        logger.error(f"Validation error in mitigate: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Unexpected error in mitigate: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error during mitigation")
