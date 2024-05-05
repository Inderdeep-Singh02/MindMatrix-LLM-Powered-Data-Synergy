from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from application.main.utils.models.model_pipe import BaseModel

class ClassificationModels(BaseModel):
    def __init__(self):
        models = [
            {'model': RandomForestClassifier(), 'param_grid': {'n_estimators': [50, 100], 'max_depth': [None, 10]}},
            {'model': GradientBoostingClassifier(), 'param_grid': {'n_estimators': [50]}},
            # {'model': LogisticRegression(solver='liblinear', max_iter=2000, dual=False), 'param_grid': {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}},
            {'model': DecisionTreeClassifier(), 'param_grid': {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5]}},
            # {'model': SVC(), 'param_grid': {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}},
            {'model': KNeighborsClassifier(), 'param_grid': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}},
            {'model': GaussianNB(), 'param_grid': {}},
            # {'model': MLPClassifier(), 'param_grid': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh'], 'alpha': [0.0001, 0.001, 0.01]}},
            {'model': AdaBoostClassifier(), 'param_grid': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}},
            {'model': LinearDiscriminantAnalysis(), 'param_grid': {}}
        ]
        super().__init__(models, classification_flag=True)

