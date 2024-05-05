from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Body
from fastapi.responses import JSONResponse
from typing import List
from application.main.utils.models.model_pipe import model_pipe
# from application.main.utils.dataframe import dataframe
from application.main.utils.meta_data import extract_metadata
from application.main.utils.preprocessing.pre_pipe import preprocessing_pipe
import pandas as pd


router = APIRouter(prefix="/model")

async def dataframe(csv_data: List[List[str]]):
    columns = csv_data[0]
    data = csv_data[1:]
    df = pd.DataFrame(data, columns=columns)
    return df


@router.post('/')
async def model(csv_data: List[List]): #csv_data: List[List]csvFile: UploadFile=File(...) target: int=Form(-1) metadata: Metadata = Body(...)
    try:
        df = await dataframe(csv_data)
        meta_data = extract_metadata(df)
        # df = await preprocessing_pipe(df, target)     
        result = await model_pipe(df, meta_data)

        return JSONResponse(content={"result": result})
    
    except HTTPException as e:
        return JSONResponse(content={"error_message": str(e.detail)})
    