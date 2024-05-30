import re
from application.main.utils.llm.llm import BaseAgent
import pandas as pd
    
# prompt_template_type = """

#     Given the metadata of a dataset, determine whether the target column {x} is intended for classification or regression analysis. Please consider the unique values of the column and/or the semantic meaning of the column name.

#     Meta-data:
#     {y}

#     Do not provide any explanation, only provide the answer Classification or Regression.
#     """

prompt_template_type = """

    Given the metadata of a dataset, determine whether the target column {x} is intended for classification or regression analysis. Please consider the unique values of the column and/or the semantic meaning of the column name.

    Meta-data:
    {y}

    Unique values of the target column and its count:
    unique values: {a},
    count of the values: {b}

    
    1. Understand the target column:
        - Check if the values are discrete or continuous.
        
    2. If the values are discrete:
        - Determine if they represent categories or classes.
        - Check if there are a small number of unique values, suggesting classification.
        
    3. If the values are continuous:
        - Consider if they represent a range or scale.
        - Evaluate if the column name implies a quantitative measurement.
        
    4. Analyze any additional context provided in the metadata.

    Do not provide explaination only the answer using classification or regression or cannot determine
    
    """

ANSWER_KEY_PATTERN = "classification or regression or cannot determine"


async def analyse_target(df, meta_data):
    target_column = df.iloc[:, -1]
    
    target_column_str = target_column.to_string(index=False) if isinstance(target_column, pd.DataFrame) else str(target_column)
    meta_data_str = meta_data.to_string(index=False) if isinstance(meta_data, pd.DataFrame) else str(meta_data)

    target_unique_values = df.iloc[:, -1].unique()
    target_unique_count = len(target_unique_values)

    input_var = {"x": target_column_str,"y": meta_data_str, "a": target_unique_values, "b": target_unique_count}
    # print(target_column_str, meta_data_str, target_unique_values, target_unique_count)


    llm = BaseAgent()
    model = llm.load_openrouter_chat_model("openai/gpt-4-turbo") #nousresearch/nous-capybara-7b:free "google/gemma-7b-it:free" "openai/gpt-3.5-turbo-16k" "openai/gpt-4o
    input_variables = ['x', 'y', 'a', 'b']
    prompt_template = await llm.load_prompt_template(prompt=prompt_template_type, input_variables=input_variables)
    output = await llm.run_chat_model_instruct(chat_model=model, prompt_template=prompt_template, input_variables=input_var)
    output = output.lower()
    print("Found_llm:", output)

    pattern = r"\b(classification|regression)\b"
    match = re.search(pattern, output)

    if match:
        if match.group(0) == "classification":
            return True
        else: 
            return False
    else:
        # print("Entered analyse")
        num_unique_values = target_column.nunique()
        threshold = 10
        if num_unique_values > threshold:
            return False
        else:
            return True
        
