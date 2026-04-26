import logging
import sys
from app.core.config import settings

def setup_logging():
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    
    # Optional: disable overly verbose third-party logs
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
logger = logging.getLogger("fairsight")
