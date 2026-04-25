from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(
    title="FairSight Core",
    description="Dataset-level bias analysis for uploaded CSV files.",
    version="1.0.0",
)

app.include_router(router)


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "service": "FairSight Core"}
