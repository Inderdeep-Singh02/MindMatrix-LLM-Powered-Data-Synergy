from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
from typing import List


class BaseModel:
    def __init__(self, models, classification_flag):
        self.models = models
        self.classification_flag = classification_flag
        self._scaler = StandardScaler() if not classification_flag else None
        self.evaluation_results = []

    async def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        performance_metric = accuracy_score if self.classification_flag else r2_score
        performance = performance_metric(y_test, y_pred)
        result = {
            'model_name': self.model.__class__.__name__,
            'best_parameters': self.best_parameters,
            'performance': performance
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

            result = await self.evaluate_model(X_test, y_test)
            results.append(result)
            # print(f"Best parameters for {model.__class__.__name__}: {self.best_parameters}")
            # print(f"Performance on the test set: {result['performance']}")
            # print("-----")

        return results


async def model_pipe(df): 
    classification_flag = False
    from application.main.utils.models.ClassificationModels import ClassificationModels
    from application.main.utils.models.RegressionModels import RegressionModels 

    if classification_flag:
        models_instance = ClassificationModels()
    else:
        models_instance = RegressionModels()

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []
    for model_info in models_instance.models:
        print(f"\nTraining and evaluating {model_info['model'].__class__.__name__}")
        model_instance = BaseModel([model_info], classification_flag)
        results.extend(await model_instance.perform_grid_search(X_train, y_train, X_test, y_test))

    return results

