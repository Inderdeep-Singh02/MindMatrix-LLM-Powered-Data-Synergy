# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from application.main.utils.preprocessing.outlier import remove_outliers

# def regression_preprocessing(df, target=None):
#     categorical_columns = df.select_dtypes(include=['object']).columns
#     label_encoded_columns = []

#     for column in categorical_columns:
#         label_encoder = LabelEncoder()
#         df[column] = label_encoder.fit_transform(df[column])
#         label_encoded_columns.append(column)

#     df.drop_duplicates(inplace=True)
#     df.fillna(df.mean(), inplace=True)
    
#     if isinstance(target, str):
#         target_column = target
#         target_index = df.columns.get_loc(target_column)
#     elif isinstance(target, int):
#         target_index = target
#         target_column = df.columns[target_index]
#     else:
#         raise ValueError("Invalid target specification. Provide either column name or index.")


#     numeric_columns = df.select_dtypes(exclude=['object']).columns.difference(label_encoded_columns).difference([target_column])

#     if not numeric_columns.empty:
#         df[numeric_columns] = StandardScaler().fit_transform(df[numeric_columns])

#     result_df = pd.DataFrame(df, columns=df.columns)
#     result_df[target_column] = df[target_column].values if target_column is not None else None
    
#     preprocessed_df = remove_outliers(result_df, threshold=5.0)

#     return preprocessed_df


import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import re
from application.main.utils.preprocessing.remove_outlier import remove_outliers

pd.options.mode.chained_assignment = None


def calcsqt(x):
    t = x.split('-')
    if len(t) == 2:
        return (float(t[0]) + float(t[1])) / 2
    try:
        return float(x)
    except ValueError:
        return None

    

def detect_float_columns(df, threwshold=0.9):
    float_columns = []
    for column in df.columns:
        count = sum(df[column].apply(lambda x: isinstance(x, str) and re.match(r'^\d+\s*-\s*\d+$', x) is not None))
        if count > 10:
            float_columns.append(column)

    return float_columns

def impute_missing_values(df):
    numeric_columns = df.select_dtypes(exclude=['object']).columns
    numeric_imputer = SimpleImputer(strategy='mean')
    df[numeric_columns] = numeric_imputer.fit_transform(df)
    df[numeric_columns] = df[numeric_columns].apply(lambda x: x.astype(str).str.split('-', expand=True).astype(float).mean(axis=1))

    categorical_columns = df.select_dtypes(include=['object']).columns
    label_encoded_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = label_encoded_imputer.fit_transform(df[categorical_columns])

    return df

def preprocess_float_columns(df, float_columns):
    for column in float_columns:
        df[column] = df[column].apply(lambda x: calcsqt(x) if isinstance(x, str) else x)
    return df


def regression_preprocessing(df, target=None):

    df.drop_duplicates(inplace=True)
    # df = impute_missing_values(df)

    if not df.dropna().empty:
        df = df.dropna()

    float_columns = detect_float_columns(df) 
    df = preprocess_float_columns(df, float_columns)

    label_encoded_columns = []

    unique_threshold = len(df) * 0.04

    for column in df.select_dtypes(exclude=['object']).columns:
        if (
            df[column].nunique() <= unique_threshold and
            df[column].dtype == 'int64'
        ):
            label_encoded_columns.append(column)

    categorical_columns = df.select_dtypes(include=['object']).columns

    for column in categorical_columns:
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column])
        label_encoded_columns.append(column)

    if isinstance(target, str):
        target_column = target
        target_index = df.columns.get_loc(target_column)
    elif isinstance(target, int):
        target_index = target
        target_column = df.columns[target_index]
    else:
        raise ValueError("Invalid target specification. Provide either column name or index.")
    

    numeric_columns = df.select_dtypes(exclude=['object']).columns.difference(label_encoded_columns).difference([target_column])
    df = remove_outliers(df, threshold=5.0)

    # print("num: ",len(numeric_columns))
    # print(numeric_columns)

    if not numeric_columns.empty:
        df[numeric_columns] = StandardScaler().fit_transform(df[numeric_columns])
    
    result_df = pd.DataFrame(df, columns=df.columns)
    result_df[target_column] = df[target_column].values if target_column is not None else None
    result_df = result_df.dropna()
    

    return result_df

