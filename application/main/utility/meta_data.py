
def extract_metadata(df):
    metadata = {
        'columns': list(df.columns),
        'data_types': df.dtypes.to_dict(),
        'null_values': df.isnull().sum().to_dict(),
        'unique_values': df.nunique().to_dict(),
        'row_count': len(df),
        'column_count': len(df.columns),
    }
    return metadata
