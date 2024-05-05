from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, Perceptron
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from application.main.utils.models.model_pipe import BaseModel

class RegressionModels(BaseModel):
    def __init__(self):
        models = [
            {'model': RandomForestRegressor(), 'param_grid': {'n_estimators': [50, 100], 'max_depth': [None, 10]}},
            {'model': GradientBoostingRegressor(), 'param_grid': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}},
            {'model': LinearRegression(), 'param_grid': {}},
            {'model': DecisionTreeRegressor(), 'param_grid': {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5]}},
            {'model': Lasso(), 'param_grid': {'alpha': [0.01, 0.1, 1.0], 'fit_intercept': [True, False]}},
            {'model': Ridge(), 'param_grid': {'alpha': [0.01, 0.1, 1.0], 'fit_intercept': [True, False]}},
            # {'model': Perceptron(),'param_grid': {'penalty': ['l2', 'l1', 'elasticnet'],'alpha': [0.0001, 0.001, 0.01],'max_iter': [100, 200, 300],'tol': [1e-3, 1e-4, 1e-5]}},
            {'model': ElasticNet(), 'param_grid': {'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.25, 0.5, 0.75], 'fit_intercept': [True, False]}},
            {'model': KNeighborsRegressor(), 'param_grid': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}},
            {'model': AdaBoostRegressor(), 'param_grid': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}} 
            ]
        super().__init__(models, classification_flag=False)
    