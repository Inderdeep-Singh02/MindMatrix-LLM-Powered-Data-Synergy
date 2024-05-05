from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from application.main.utils.dataframe import dataframe
from application.main.utils.preprocessing.pre_pipe import preprocessing_pipe

router = APIRouter(prefix='/preprocess')

@router.post('/')
async def preprocess( csvFile: UploadFile=File(...), target:int=Form(-1)):
    try:
        df = await dataframe(csvFile=csvFile)
        preprocessed_df =  await preprocessing_pipe(df=df, target=target)
        preprocessed_list = [preprocessed_df.columns.tolist()] + preprocessed_df.values.tolist()
        return JSONResponse(content=preprocessed_list)
    except HTTPException as e:
        return JSONResponse(content={"error_message": str(e.detail)})
    


    