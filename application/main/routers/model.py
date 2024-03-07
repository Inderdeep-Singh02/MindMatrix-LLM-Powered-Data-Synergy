from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/model")

@router.post('/')
async def model(csvFile: UploadFile = File(...)):

    return JSONResponse(content={"mes": "age"})


