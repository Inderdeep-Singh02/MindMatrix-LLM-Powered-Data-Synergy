import json
def extract_metadata(df):
    metadata = {
        "columns": list(df.columns),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_values": df.isnull().sum().to_dict(),
        "unique_values": df.nunique().to_dict(),
        "row_count": len(df),
        "column_count": len(df.columns),
    }
    return json.dumps(metadata)