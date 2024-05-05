import re
from application.main.utils.llm.llm import BaseAgent
import pandas as pd
    
prompt_template_type = """

    Given the metadata of a dataset, determine whether the target column {x} is intended for classification or regression analysis. Please consider the unique values of the column and/or the semantic meaning of the column name.

    Meta-data:
    {y}

    Do not provide any explanation, only provide the answer Classification or Regression.
    """

async def analyse_target(df, meta_data):
    target_column = df.iloc[:, -1]

    target_column_str = target_column.to_string(index=False) if isinstance(target_column, pd.DataFrame) else str(target_column)
    meta_data_str = meta_data.to_string(index=False) if isinstance(meta_data, pd.DataFrame) else str(meta_data)

    input_var = {"x": target_column_str,"y": meta_data_str}


    llm = BaseAgent()
    model = llm.load_openrouter_chat_model("google/gemma-7b-it:free") #nousresearch/nous-capybara-7b:free
    input_variables = ['x', 'y']
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
        
