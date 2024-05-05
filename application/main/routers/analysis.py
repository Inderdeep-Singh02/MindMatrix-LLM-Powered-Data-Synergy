from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, File, UploadFile, HTTPException

from application.main.utils.sweet_visual import sweet_visual
from application.main.utils.meta_data import extract_metadata
from application.main.utils.dataframe import dataframe


router = APIRouter(prefix='/analysis')
templates = Jinja2Templates(directory="./application/main/static/templates")

@router.post("/")
async def visual(csvFile: UploadFile = File(...)):
    try:
        df = await dataframe(csvFile=csvFile)
        report_html = await sweet_visual(df)
        meta_data = extract_metadata(df)

        response_data = {
            "report_html": report_html,
            "meta_data": meta_data
        }

        return JSONResponse(content=response_data)
    except HTTPException as e:
        return JSONResponse(content={"error_message": str(e.detail)})
