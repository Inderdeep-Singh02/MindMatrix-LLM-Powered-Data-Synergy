from fastapi import APIRouter, File, UploadFile, Request, HTTPException, Form
from fastapi.responses import JSONResponse
from application.main.utils.dataframe import dataframe
from application.main.utils.preprocessing import analyse_target 
from application.main.utils.preprocessing.regression.regresion import regression_preprocessing 

router = APIRouter(prefix='/preprocess')

@router.post('/')
async def preprocess(request: Request, csvFile: UploadFile = File(...), target: int = Form(-1)):
    try:
        df = await dataframe(csvFile=csvFile)
        preprocessed_df = regression_preprocessing(df, target)
        # preprocessed_dict = preprocessed_df.to_dict(orient='records')
        preprocessed_list = [preprocessed_df.columns.tolist()] + preprocessed_df.values.tolist()
        return JSONResponse(content=preprocessed_list)
    except HTTPException as e:
        return JSONResponse(content={"error_message": str(e.detail)})
    



    