from fastapi import HTTPException
import pandas as pd
import io
import csv
from io import StringIO

# async def dataframe(csvFile):  
#     if not csvFile.filename.endswith(".csv"):
#             raise HTTPException(status_code=400, detail="Only CSV files are allowed")
#     content = await csvFile.read()
#     df = pd.read_csv(io.BytesIO(content))
#     return df


# async def read_csv_with_fallback(file_like_object, delimiter, quotechar):
#     try:
#         df = pd.read_csv(file_like_object, delimiter=delimiter, quotechar=quotechar)
#     except pd.errors.ParserError:
#         raise HTTPException(status_code=400, detail="Failed to parse CSV file")

#     return df

async def read_csv_with_fallback(file_like_object, delimiter, quotechar='"'):
    try:
        df = pd.read_csv(file_like_object, sep=delimiter, quotechar=quotechar)
        return df
    except pd.errors.ParserError:
        fallback_delimiter = ','
        df = pd.read_csv(file_like_object, sep=fallback_delimiter, quotechar=quotechar)
        return df

async def dataframe(csvFile):  
    if not csvFile.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    content = await csvFile.read()


    content_str = content.decode('utf-8')
    delimiter = csv.Sniffer().sniff(content_str[:1024]).delimiter
    file_like_object = StringIO(content_str)
    df = await read_csv_with_fallback(file_like_object, delimiter, quotechar='"')
        # df = pd.read_csv(io.BytesIO(content), chunksize=4096)
        # df = pd.concat(df, ignore_index=True)
        # print(df)


    return df