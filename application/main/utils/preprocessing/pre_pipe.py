import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler ,MinMaxScaler
import re
from application.main.utils.preprocessing.remove_outlier import remove_outliers
from application.main.utils.preprocessing.impute_value import impute_missing_values
from application.main.utils.llm.llm import BaseAgent
import string  
import numpy as np 

# prompt_template_SM = """

#     Given the decription of dataset, determine whether the numeric columns are to be normalized or standardized.

#     Description:
#     {a}

#     Do not provide any explanation, only provide the answer Standardization or Normalization.
#     """

# TODO: prompt for preprocess explaination on columns along with normalization 

prompt_template_SM = """
    Given the description of the dataset, determine whether the numeric columns are to be normalized or standardized.

    1. Evaluate the range (difference between max and min values) of the numeric data.
    2. Assess the standard deviation relative to the mean for each numeric column.
    3. If the range is large and/or the standard deviation is high compared to the mean, prefer normalization.
    4. If the range is moderate and the data seems to follow a Gaussian distribution, prefer standardization.
    5. Make a decision based on the above evaluations.

    Description:
    {a}

    Do not provide any explanation, only provide the answer Standardization or Normalization.
"""

prompt_template_UC = """
Given the unique values and their counts for each column in the dataset, identify the special symbols that should be considered as missing values (NA/null).

1. Examine the unique values and their counts for each column.
2. Identify values that are non-numeric or seem out of place in the context of the column.
3. Determine which of these values should be replaced with NA/null.
4. Make a decision based on the above evaluations.

Unique Values and Counts:
{unique_values}

Do not provide any explanation, only list the symbols that should be considered as missing values, separated by commas.

"""

prompt_template_RI = """
Given the column names in the dataset, identify any columns that are likely identifiers (such as ID, serial number, S.no, or index) that can be removed.

Column Names:
{column_names}

Do not provide any explanation, only list the columns that should be considered as identifiers and removed, separated by commas.

Chain of Thought:
1. Examine the column names and identify any that suggest they are unique identifiers (e.g., ID, SerialNo, S.no, Index).
2. List these columns as they are typically used for identification and not for analysis.

Do not provide any explanation, only the name of the column else Cannot Determine

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

# def replace_special_symbols_with_nan(df, symbols):
#     for symbol in symbols:
#         if not isinstance(symbol, (int, float, complex)) and not str(symbol).isdigit():
#             try:
#                 df.replace(symbol, np.nan, inplace=True)
#             except Exception as e:
#                 print(f"Error replacing {symbol}: {e}")

def replace_special_symbols_with_nan(df, symbols):
    for symbol in symbols:
        if not isinstance(symbol, (int, float, complex)) and not str(symbol).isdigit():
            try:
                print("Replacing", symbol)
                if not (symbol.startswith(' -') and symbol.lstrip(' -').isdigit()):
                    df.replace(symbol, np.nan, inplace=True)
            except Exception as e:
                print(f"Error replacing {symbol}: {e}")

def get_unique_counts(df):
    unique_values = {}
    for column in df.columns:
        unique_values[column] = df[column].value_counts().to_dict()
    return unique_values



def get_special_symbol_counts(df):
    special_symbol_counts = {}
    
    special_characters = set(string.punctuation)  
    
    def contains_url_or_long_non_symbolic(column):
        for value in df[column]:
            if isinstance(value, str):
                if 'http://' in value or 'https://' in value:
                    return True
                if len(value) > 4 and any(char in special_characters for char in value):
                    return True
        return False

    def is_special_symbol(x):
        if pd.api.types.is_number(x):  
            return False
        elif isinstance(x, str):
            return any(char in special_characters and not x.startswith('"') and not x.isdigit() for char in x)
        else:
            return False
    
    for column in df.columns:
        if contains_url_or_long_non_symbolic(column):
            continue
        
        special_symbol_values = df[column].apply(lambda x: x if is_special_symbol(x) else np.nan)
        
        counts = special_symbol_values.value_counts(dropna=True).to_dict()
        
        counts.pop(np.nan, None)
        
        special_symbol_counts[column] = counts
        
    return special_symbol_counts

def remove_identifier_columns(df, column):
    for column in column:
        if column in df.columns:
            df.drop(column, axis=1, inplace=True)
        else:
            print(f"Warning: Column '{column}' not found in DataFrame")

def get_column_names(df):
    return df.columns.tolist()

async def preprocessing_pipe(df, target=None):

    if isinstance(target, str):
        target_column = target
        target_index = df.columns.get_loc(target_column)
    elif isinstance(target, int):
        target_index = target
        target_column = df.columns[target_index]
    else:
        raise ValueError("Invalid target specification. Provide either column name or index.")
    df = pd.DataFrame(df, columns=df.columns)
    df[target_column] = df[target_column].values if target_column is not None else None

    df = df[[col for col in df.columns if col != target_column] + [target_column]]
    
    LLM_flag = True
    try:
        llm = BaseAgent()
        model = llm.load_openrouter_chat_model("openai/gpt-4-turbo") # "nousresearch/nous-capybara-7b:free" "openai/gpt-3.5-turbo-16k" "openai/gpt-4o"
    except:
        LLM_flag = False

    column_names = get_column_names(df)
    print(column_names)
    input_variables = ['column_names']
    input_var = {"column_names": column_names}
    prompt_template = await llm.load_prompt_template(prompt=prompt_template_RI, input_variables=input_variables)
    output = await llm.run_chat_model_instruct(chat_model=model, prompt_template=prompt_template, input_variables=input_var)
    output = output.replace(" ", "").split(",")
    print("LLM out for IDentifier: ",output)
    

    remove_identifier_columns(df, output)
    float_columns = detect_float_columns(df) 
    
    
    if df.isnull().values.any(): 
        df = preprocess_float_columns(df, float_columns)
    
    unique_counts = get_special_symbol_counts(df)
    print(unique_counts)

    input_variables = ['unique_values']
    input_var = {"unique_values": unique_counts}

    prompt_template = await llm.load_prompt_template(prompt=prompt_template_UC, input_variables=input_variables)
    output = await llm.run_chat_model_instruct(chat_model=model, prompt_template=prompt_template, input_variables=input_var)
    output = output.lower()
    print("LLM out for special: ", output.replace(" ", ""))

    special_symbols = list(set(output.split(",")))
    print(special_symbols)
    replace_special_symbols_with_nan(df, special_symbols)
    
    
    desc_df = df.describe()
    if df.isnull().values.any(): 
        df = impute_missing_values(df)    

    
    df = drop_null_columns(df)
    df = df.drop_duplicates()

   
            

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

    # if isinstance(target, str):
    #     target_column = target
    #     target_index = df.columns.get_loc(target_column)
    # elif isinstance(target, int):
    #     target_index = target
    #     target_column = df.columns[target_index]
    # else:
    #     raise ValueError("Invalid target specification. Provide either column name or index.")
    numeric_columns = df.select_dtypes(exclude=['object']).columns.difference(label_encoded_columns).difference([target_column])
    df = remove_outliers(df, threshold=5.0)


    # if not numeric_columns.empty:
    #     df[numeric_columns] = StandardScaler().fit_transform(df[numeric_columns])

    input_var = {"a": desc_df}
    print(desc_df)

    # llm = BaseAgent()
    # model = llm.load_openrouter_chat_model("openai/gpt-3.5-turbo-16k") # "nousresearch/nous-capybara-7b:free"
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
    
        
    # result_df = pd.DataFrame(df, columns=df.columns)
    # result_df[target_column] = df[target_column].values if target_column is not None else None
    # result_df = result_df[[col for col in result_df.columns if col != target_column] + [target_column]]
    result_df = df.dropna()

    
    return result_df