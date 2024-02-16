from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Request, File, UploadFile, HTTPException
import pandas as pd
import io

from application.main.utility.sweet_visual import sweetVisual
from application.main.utility.meta_data import extract_metadata


router = APIRouter(prefix='/EdaVis')
templates = Jinja2Templates(directory="./application/main/static/templates")

@router.get('/', response_class=HTMLResponse)
async def EdaVis(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@router.post("/")
async def visual(request: Request, csvFile: UploadFile = File(...)):
    if not csvFile.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    content = await csvFile.read()
    df = pd.read_csv(io.BytesIO(content))
    report_html = await sweetVisual(df)
    meta_data = extract_metadata(df)
    print(meta_data)

    return templates.TemplateResponse("dashboard.html", {"request": request, "report_html": report_html})
