from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from application.main.utils.models.model_pipe import model_pipe
from application.main.utils.dataframe import dataframe
from application.main.utils.preprocessing.pre_pipe import preprocessing_pipe

router = APIRouter(prefix="/model")

@router.post('/')
async def model(csvFile: UploadFile=File(...), target: int=Form(-1)):
    try:
        df = await dataframe(csvFile=csvFile)
        preprocessed_df = preprocessing_pipe(df, target)
        result = await model_pipe(preprocessed_df)
        return JSONResponse(content={"result": result})
    
    except HTTPException as e:
        return JSONResponse(content={"error_message": str(e.detail)})