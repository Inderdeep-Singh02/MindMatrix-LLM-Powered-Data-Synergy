from fastapi.responses import JSONResponse
from fastapi.routing import APIRouter

router = APIRouter(prefix='/health-check')

@router.get('/')
async def healthcheck():
    return JSONResponse(content={"message": "MindMatrix"}, status_code=200)
