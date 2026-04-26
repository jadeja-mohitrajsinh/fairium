import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    GOOGLE_API_KEY: str = ""
    BACKEND_HOST: str = "127.0.0.1"
    BACKEND_PORT: int = 8001
    
    LOG_LEVEL: str = "INFO"
    
    # File Upload Limits
    MAX_UPLOAD_SIZE_MB: int = 50
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
