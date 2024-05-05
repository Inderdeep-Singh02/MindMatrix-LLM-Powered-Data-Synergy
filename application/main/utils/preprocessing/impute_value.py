from sklearn.impute import SimpleImputer
import pandas as pd


def impute_missing_values(df):
    numeric_columns = df.select_dtypes(include=['number']).columns
    object_columns = df.select_dtypes(include=['object']).columns

    if not numeric_columns.empty:
        numeric_columns_with_data = df[numeric_columns].notnull().any()
        numeric_columns = numeric_columns[numeric_columns_with_data]
        
        if not numeric_columns.empty:
            numeric_columns_with_nan = df[numeric_columns].isnull().all()
            numeric_columns = numeric_columns[~numeric_columns_with_nan]
            
            if not numeric_columns.empty:
                imputer_numeric = SimpleImputer(strategy='mean')
                df[numeric_columns] = imputer_numeric.fit_transform(df[numeric_columns])
    if not object_columns.empty:
        imputer_object = SimpleImputer(strategy='most_frequent')
        df[object_columns] = imputer_object.fit_transform(df[object_columns])

    return df