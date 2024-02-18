from fastapi import HTTPException
import pandas as pd
import io

async def dataframe(csvFile):  
    if not csvFile.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    content = await csvFile.read()
    df = pd.read_csv(io.BytesIO(content))
    return df