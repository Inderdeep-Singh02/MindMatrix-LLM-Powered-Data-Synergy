from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Request, File, UploadFile, HTTPException

from application.main.utils.sweet_visual import sweetVisual
from application.main.utils.meta_data import extract_metadata
from application.main.utils.dataframe import dataframe


router = APIRouter(prefix='/meta_visual')
templates = Jinja2Templates(directory="./application/main/static/templates")

@router.get('/', response_class=HTMLResponse)
async def EdaVis(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@router.post("/")
async def visual(request: Request, csvFile: UploadFile = File(...)):
    try:
        df = await dataframe(csvFile=csvFile)
        report_html = await sweetVisual(df)
        meta_data = extract_metadata(df)
        print(meta_data)

        return templates.TemplateResponse("dashboard.html", {"request": request, "report_html": report_html, "error_message": None})
    except HTTPException as e:
        return templates.TemplateResponse(
            "dashboard.html",
            {"request": request, "report_html": None, "error_message": str(e.detail)}
        )