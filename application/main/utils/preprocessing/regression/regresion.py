import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from application.main.utils.preprocessing.outlier import remove_outliers

def reg_preprocessing(df, target=None):
    categorical_columns = df.select_dtypes(include=['object']).columns
    label_encoded_columns = []

    for column in categorical_columns:
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column])
        label_encoded_columns.append(column)

    df.drop_duplicates(inplace=True)
    df.fillna(df.mean(), inplace=True)

    if isinstance(target, str):
        target_column = target
        target_index = df.columns.get_loc(target_column)
    elif isinstance(target, int):
        target_index = target
        target_column = df.columns[target_index]
    else:
        raise ValueError("Invalid target specification. Provide either column name or index.")


    numeric_columns = df.select_dtypes(exclude=['object']).columns.difference(label_encoded_columns)
    df[numeric_columns] = StandardScaler().fit_transform(df[numeric_columns])

    result_df = pd.DataFrame(df, columns=df.columns)
    result_df[target_column] = df[target_column].values if target_column is not None else None
    
    preprocessed_df = remove_outliers(result_df, threshold=5.0)

    return preprocessed_df


