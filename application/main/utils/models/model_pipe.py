from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error, confusion_matrix 
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib
from sklearn.preprocessing import StandardScaler
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import io
import json
import base64   
import numpy as np
import re
import pandas as pd
import random
from application.main.utils.preprocessing.analyse_target import analyse_target
from application.main.utils.llm.llm import BaseAgent

matplotlib.use('Agg')


prompt_template_model = """
For the given metadata of a dataset:
meta_data: '{y}'

Which model is more suitable from the given pool of models:
models: '{z}'

Provide 5 names of the most suitable model based on the provided metadata and the pool of models. Do not include explanations, just the model name.
"""

# prompt_template_model = """
#     Consider the following scenario: You have metadata for a dataset, provided as '{y}', and a pool of models available, listed as '{z}'. Your task is to determine the most suitable model from this pool based on the provided metadata.

#     Without providing explanations, please generate 5 names of the models that are best suited to the given dataset.
# """

# TODO: add reflection

def format_floats(obj):
        if isinstance(obj, float):
            return f"{obj:.2f}"
        elif isinstance(obj, list):
            return [format_floats(item) for item in obj]
        else:
            return obj
        
class BaseModel:
    def __init__(self, models, classification_flag):
        self.models = models
        self.classification_flag = classification_flag
        self._scaler = StandardScaler() if not classification_flag else None
        self.evaluation_results = []

    
    

    async def evaluate_model(self, X_train, X_test, y_test):
        y_pred = self.model.predict(X_test)
        # performance_metrics = {
        #     'accuracy': accuracy_score,
        #     'r2': r2_score,
        #     'mse': mean_squared_error,
        #     'mae': mean_absolute_error
        # }
        performance_metrics = {
        'accuracy': accuracy_score,
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score,
        'r2': r2_score,
        'mse': mean_squared_error,
        'mae': mean_absolute_error
    }
        
        # if self.classification_flag:
        #     performance_metrics.pop('r2')  
        # else:
        #     performance_metrics.pop('accuracy') 
        if self.classification_flag:
            performance_metrics.pop('r2')
            performance_metrics.pop('mse')
            performance_metrics.pop('mae')
        else:
            performance_metrics.pop('accuracy')
            performance_metrics.pop('f1')
            performance_metrics.pop('precision')
            performance_metrics.pop('recall') 

        # evaluation_results = {}
        # for metric_name, metric_func in performance_metrics.items():
        #     if metric_name == 'mse' or metric_name == 'mae':
        #         performance = metric_func(y_test, y_pred)
        #     else:
        #         performance = metric_func(y_test, y_pred)
                    
        #     evaluation_results[metric_name] = performance

        evaluation_results = {}
        # for metric_name, metric_func in performance_metrics.items():
        #     performance = metric_func(y_test, y_pred)
        #     evaluation_results[metric_name] = f"{performance:.3f}" if isinstance(performance, float) else performance
        # evaluation_results = {}
        for metric_name, metric_func in performance_metrics.items():
            if metric_name in ['f1', 'precision', 'recall']:
                performance = metric_func(y_test, y_pred, average='weighted')
            else:
                performance = metric_func(y_test, y_pred)
            evaluation_results[metric_name] = f"{performance:.3f}" if isinstance(performance, float) else performance
        
        try:
            if self.classification_flag:
                performance_graph = self.generate_confusion_matrix(y_test, y_pred)
            else:
                performance_graph = self.generate_actual_vs_prediction_graph(X_train, X_test, y_test, y_test, y_pred)
        except Exception as e:
            pass

        result = {
            'model_name': self.model.__class__.__name__,
            'best_parameters': self.best_parameters,
            'performance': evaluation_results,
            'performance_graph': performance_graph,
            'cv_results_dict': json.loads(self.res)
        }

        self.evaluation_results.append(result)
        return result
    
    async def perform_grid_search(self, X_train, y_train, X_test, y_test) -> List[dict]:
        results = []
        for model_info in self.models:
            model = model_info['model']
            param_grid = model_info['param_grid']
            grid_search = GridSearchCV(
                model, param_grid, cv=KFold(n_splits=5),
                scoring='accuracy' if self.classification_flag else 'r2', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            self.model = grid_search.best_estimator_
            self.best_parameters = grid_search.best_params_
            
            # self.res = json.dumps({key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in grid_search.cv_results_.items()})
            
            # Format all float values in the dictionary to two decimal places
            formatted_results = {
            key: format_floats(value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in grid_search.cv_results_.items()
            }

            self.res = json.dumps(formatted_results, indent=4)


            result = await self.evaluate_model(X_train, X_test, y_test)
            results.append(result)

        return results

    def generate_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f"{self.model.__class__.__name__} Confusion Matrix")
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        performance_graph = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return performance_graph

   
    def generate_actual_vs_prediction_graph(self, X_train, X_test, y_train, y_test, y_pred):
        
        plt.figure(figsize=(10,10))
        plt.scatter(y_test, y_pred, c='crimson')
        plt.yscale('log')
        plt.xscale('log')

        p1 = max(max(y_pred), max(y_test))
        p2 = min(min(y_pred), min(y_test))
        plt.plot([p1, p2], [p1, p2], 'b-')
        plt.xlabel('True Values', fontsize=15)
        plt.ylabel('Predictions', fontsize=15)
        plt.axis('equal')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        performance_graph = base64.b64encode(buffer.getvalue()).decode()
        
        plt.close()

        return performance_graph


async def model_pipe(df, meta_data): 
    classification_flag = await analyse_target(df, meta_data=meta_data)

    
    from application.main.utils.models.ClassificationModels import ClassificationModels
    from application.main.utils.models.RegressionModels import RegressionModels 

    if classification_flag:
        models_instance = ClassificationModels()
    else:
        models_instance = RegressionModels()
    
    meta_data_str = meta_data.to_string(index=False) if isinstance(meta_data, pd.DataFrame) else str(meta_data)
    input_var = {'y': meta_data_str, 'z': models_instance.models}

    llm = BaseAgent()
    model = llm.load_openrouter_chat_model("openai/gpt-4-turbo") #"meta-llama/llama-3-8b-instruct:free" "google/gemma-7b-it:free"
    input_variables = ['y', 'z']
    prompt_template = await llm.load_prompt_template(prompt=prompt_template_model, input_variables=input_variables)
    output = await llm.run_chat_model_instruct(chat_model=model, prompt_template=prompt_template, input_variables=input_var)
    output = output.lower()
    print("models:", output)
    
    model_pool = [model_info['model'].__class__.__name__.lower() for model_info in models_instance.models]
    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(model) for model in model_pool) + r')\b')

    matches = pattern.findall(output)
    model_list = [re.sub(r'[^a-zA-Z0-9]+', '', match) for match in matches]
    model_list=[]

    if len(model_list) < 5:
        remaining_models = random.sample([model for model in model_pool if model not in model_list], 5 - len(model_list))
        # remaining_models = model_pool[:5]
        model_list += remaining_models
    elif len(model_list) > 5:
        model_list = model_list[:5]
 

    print("Final model list:", model_list)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []

    for model_info in models_instance.models:
        model_name = model_info['model'].__class__.__name__
        
        if  model_name.lower() in model_list:
            print(f"\nTraining and evaluating {model_name}")
            model_instance = BaseModel([model_info], classification_flag)
            try:
                results.extend(await model_instance.perform_grid_search(X_train, y_train, X_test, y_test))
            except Exception as e:
                print(e)
                return {"error_message": str(e)}

    return results


