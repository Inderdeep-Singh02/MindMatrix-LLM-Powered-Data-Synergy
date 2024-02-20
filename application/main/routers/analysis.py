from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Request, File, UploadFile, HTTPException

from application.main.utils.sweet_visual import sweetVisual
from application.main.utils.meta_data import extract_metadata
from application.main.utils.dataframe import dataframe


router = APIRouter(prefix='/analysis')
templates = Jinja2Templates(directory="./application/main/static/templates")

# @router.get('/', response_class=HTMLResponse)
# async def EdaVis(request: Request):
#     return templates.TemplateResponse("dashboard.html", {"request": request})

# @router.post("/")
# async def visual(request: Request, csvFile: UploadFile = File(...)):
#     try:
#         df = await dataframe(csvFile=csvFile)
#         report_html = await sweetVisual(df)
#         meta_data = extract_metadata(df)
        
#         return templates.TemplateResponse("dashboard.html", {"request": request, "report_html": report_html, "error_message": None})
#     except HTTPException as e:
#         return templates.TemplateResponse(
#             "dashboard.html",
#             {"request": request, "report_html": None, "error_message": str(e.detail)}
#         )
    
@router.post("/")
async def visual(request: Request, csvFile: UploadFile = File(...)):
    try:
        df = await dataframe(csvFile=csvFile)
        report_html = await sweetVisual(df)
        meta_data = extract_metadata(df)

        response_data = {
            "report_html": report_html,
            "meta_data": meta_data
        }

        return JSONResponse(content=response_data)
    except HTTPException as e:
        return JSONResponse(content={"error_message": str(e.detail)})
