import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler ,MinMaxScaler
import re
from application.main.utils.preprocessing.remove_outlier import remove_outliers
from application.main.utils.preprocessing.impute_value import impute_missing_values
from application.main.utils.llm.llm import BaseAgent

    
prompt_template_SM = """

    Given the decription of dataset, determine whether the numeric columns are to be normalized or standardized.

    Description:
    {a}

    Do not provide any explanation, only provide the answer Standardization or Normalization.
    """

pd.options.mode.chained_assignment = None

def calc_sqt(x):
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


def preprocess_float_columns(df, float_columns):
    for column in float_columns:
        df[column] = df[column].apply(lambda x: calc_sqt(x) if isinstance(x, str) else x)
    return df

def drop_null_columns(df):
    null_columns = df.columns[df.isnull().all()]
    df.drop(columns=null_columns, inplace=True)
    return df

async def preprocessing_pipe(df, target=None):
    desc_df = df.describe()

    df = drop_null_columns(df)
    
    df = df.drop_duplicates()

    float_columns = detect_float_columns(df) 
    
    df = preprocess_float_columns(df, float_columns)
    
    if df.isnull().values.any(): 
        df = preprocess_float_columns(df, float_columns)
        df = impute_missing_values(df)
            

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

    # if not numeric_columns.empty:
    #     df[numeric_columns] = StandardScaler().fit_transform(df[numeric_columns])

    input_var = {"a": desc_df}

    llm = BaseAgent()
    model = llm.load_openrouter_chat_model("nousresearch/nous-capybara-7b:free")
    input_variables = ['a']
    prompt_template = await llm.load_prompt_template(prompt=prompt_template_SM, input_variables=input_variables)
    output = await llm.run_chat_model_instruct(chat_model=model, prompt_template=prompt_template, input_variables=input_var)
    output = output.lower()

    pattern = r"\b(standardization|normalization)\b"
    match = re.search(pattern, output)
    try:

        if match:
            print("Found:", match.group(0))
            if match.group(0) == "standardization":
                if not numeric_columns.empty:
                    df[numeric_columns] = StandardScaler().fit_transform(df[numeric_columns])
            else:
                if not numeric_columns.empty:
                    for column in numeric_columns:
                        df[column] = MinMaxScaler().fit_transform(df[[column]])
        else:
            if not numeric_columns.empty:
                df[numeric_columns] = StandardScaler().fit_transform(df[numeric_columns])
    except Exception as e:
        print(e)
        
    result_df = pd.DataFrame(df, columns=df.columns)
    result_df[target_column] = df[target_column].values if target_column is not None else None
    result_df = result_df[[col for col in result_df.columns if col != target_column] + [target_column]]
    result_df = result_df.dropna()

    return result_df