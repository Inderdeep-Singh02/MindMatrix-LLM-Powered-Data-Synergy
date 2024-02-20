from fastapi import APIRouter, File, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse
from application.main.utils.dataframe import dataframe
from application.main.utils.preprocessing import analyse_target 
from application.main.utils.preprocessing.regression.regresion import reg_preprocessing 




router = APIRouter(prefix='/preprocess')

@router.get('/')
def pp():
    return JSONResponse(content={"mmm":"jjj"}, status_code=200)


@router.post('/')
async def preprocess(request: Request, csvFile: UploadFile = File(...)):
    
    try:
        df = await dataframe(csvFile=csvFile)
        preprocessed_df = reg_preprocessing(df, target=-1)
        preprocessed_dict = preprocessed_df.to_dict(orient='records')
        return JSONResponse(content=preprocessed_dict)
    except HTTPException as e:
        return JSONResponse(content={"error_message": str(e.detail)})
    



    