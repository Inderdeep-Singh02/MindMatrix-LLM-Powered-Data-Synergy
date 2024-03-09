from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from application.main.utils.models.model_pipe import BaseModel


class ClassificationModels(BaseModel):
    def __init__(self):
        models = [
            {'model': RandomForestClassifier(), 'param_grid': {'n_estimators': [50, 100], 'max_depth': [None, 10]}},
            {'model': GradientBoostingClassifier(), 'param_grid': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}},
            # {'model': LogisticRegression(solver='liblinear', max_iter=1000, dual=False), 'param_grid': {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}}
        ]
        super().__init__(models, classification_flag=True)

