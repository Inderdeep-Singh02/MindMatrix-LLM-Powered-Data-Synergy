from sklearn.ensemble import  RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from application.main.utils.models.model_pipe import BaseModel


class RegressionModels(BaseModel):
    def __init__(self):
        models = [
            {'model': RandomForestRegressor(), 'param_grid': {'n_estimators': [50, 100], 'max_depth': [None, 10]}},
            {'model': GradientBoostingRegressor(), 'param_grid': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}},
            {'model': LinearRegression(), 'param_grid': {}}
        ]
        super().__init__(models, classification_flag=False)
    