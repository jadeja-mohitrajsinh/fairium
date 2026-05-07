from fastapi import APIRouter
from app.api.routes.analysis import router as analysis_router
from app.api.routes.mitigation import router as mitigation_router
from app.api.routes.datasets import router as datasets_router
from app.api.routes.decisions import router as decisions_router
from app.api.routes.explain import router as explain_router

router = APIRouter()
router.include_router(analysis_router)
router.include_router(mitigation_router)
router.include_router(datasets_router)
router.include_router(decisions_router)
router.include_router(explain_router)
