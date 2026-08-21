
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import make_scorer, accuracy_score, r2_score, recall_score, roc_auc_score, precision_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import pandas as pd 


class Tuner:
    '''
    Класс для полуавтоматического подбора параметров моделей
    
    Parameters
    ----------
    model : str, list, or None
        Модели для обучения: 'linear', 'rf', 'gb', 'xgboost', 'lgbm', 'svm', 'knn'
        Можно передать список или None для интерактивного выбора
    X : pd.DataFrame or np.ndarray
        Признаки
    y : pd.Series or np.ndarray
        Целевая переменная
    type_model : str
        'regression', 'classification', 'auto'
    searcher : str
        'auto', 'RandomizedSearchCV', 'GridSearchCV', None
    scoring : str or callable
        Метрика для оценки, 'auto' для выбора по умолчанию
    cv : int or 'auto'
        Количество фолдов для кросс-валидации
    n_iter : int
        Количество итераций для RandomizedSearchCV
    random_state : int
        Для воспроизводимости
    
    '''
    def __init__(self, 
                 model=None,
                 X=None,
                 y=None,
                 type_model='auto',
                 searcher='auto', 
                 scoring='auto',
                 cv='auto',
                 n_iter=20,
                 random_state=42
                 ):
        self.random_state=random_state
        self.X = X 
        self.y = y
        self.type_model = self._type_model(type_model)
        self.cv = self._cv(cv)
        self.searcher = self._searcher(searcher)
        self.model = self._input_model(model)
        self.scoring = self._scoring(scoring)
        self.n_iter = n_iter
        
        
        self.best_models_ = []    
        self.best_params_ = []
        self.best_scores_ = []
        self.results_df_ = None
        
    def _scoring(self, scoring):
        if scoring == 'auto':
            if self.type_model == 'classification':
                return make_scorer(accuracy_score)
            else:
                return make_scorer(r2_score)
    
        elif callable(scoring):
            return make_scorer(scoring)
        
        return scoring
        
    def _cv(self, cv):
        '''
        cv: int, None
        '''
        if cv is None:
            return None 
        elif cv == 'auto':
            return 5
        elif cv <= 0:
            raise ValueError(f'erorr cv={cv} (cv < 0)')
        return cv
        
    def _searcher(self, searcher):
        '''
        searcher: 'auto', 'RandomizedSearchCV', 'GridSearchCV', None
        '''
        if searcher == 'auto' and self.cv is None:
            return None            
        elif searcher == 'auto' and self.cv is not None:
            searcher = 'RandomizedSearchCV'
            
        return searcher
    
    def _type_model(self, type_model):
        '''
        type_model: 'regression', 'classification'
        '''
        if type_model == 'auto' and self.y is not None:
            type_model = 'classification' if len(set(self.y)) <= 10 else 'regression' 
        elif type_model == 'auto' and self.y is None:
            type_model = 'regression'
        elif type_model is None:
            raise ValueError(f'not avaliable type model: {type_model},\
                try: regression, classification, auto')
        
        return type_model
                 
    def _input_model(self, model):
        '''
        'linear': LinearRegression(), LogisticRegression(),\n
        'rf': RandomForestRegressor(), RandomForestClassifier(),\n
        'gb': GradientBoostingRegressor(), GradientBoostingClassifier(),\n
        'xgboost': XGBRegressor(), XGBClassifier(),\n
        'lgbm': LGBMRegressor(verbose=-1), LGBMClassifier(verbose=-1),\n
        'svm': SVR() SVC(),\n
        'knn': KNeighborsRegressor(), KNeighborsClassifier()
        '''
        models_dict = self._get_models_dict()
        
        if model is None:
            return self._choice_model()
        elif isinstance(model, list):
            return [models_dict[m] if isinstance(m, str) else m for m in model]
        elif isinstance(model, str):
            return [models_dict[model]]
        return [model]
      
    def _choice_model(self):
        models = {
            'linear': LinearRegression() if self.type_model == 'regression' else LogisticRegression(),
            'rf': RandomForestRegressor() if self.type_model == 'regression' else RandomForestClassifier(),
            'gb': GradientBoostingRegressor() if self.type_model == 'regression' else GradientBoostingClassifier(),
            'xgboost': XGBRegressor() if self.type_model == 'regression' else XGBClassifier(),
            'lgbm': LGBMRegressor(verbose=-1) if self.type_model == 'regression' else LGBMClassifier(verbose=-1),
            'svm': SVR() if self.type_model == 'regression' else SVC(),
            'knn': KNeighborsRegressor() if self.type_model == 'regression' else KNeighborsClassifier()
        }
        
        print('models:')
        print(', '.join(models.keys()))
        print(' "all" - pick all models')
        print(' "boost" - comparing boosting models(lgbm, gb, xgboost)')
        print(' "linear, rf, etc" - pick your models like list')
        
        choice = input('\nYour choice: ').strip().lower()
        
        if choice == 'all':
            selected = list(models.values())
        elif choice == 'boost':
            selected = [models['xgboost'], models['lgbm'], models['gb']]
        else:
            keys = [k.strip() for k in choice.split(',')]
            selected = [models[k] for k in keys if k in models]
        
            not_found = [k for k in keys if k not in models]
            if not_found:
                raise ValueError(f'models not found {', '.join(not_found)}')
            if not selected:
                raise ValueError('choice model property')
        
        return selected 
    
    def _get_param_grid(self, model):
        model_name = model.__class__.__name__

        if self.type_model == 'regression':
            if 'LinearRegression' in model_name:
                return {'fit_intercept': [True, False]}
            elif 'RandomForestRegressor' in model_name:
                return {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            elif 'GradientBoostingRegressor' in model_name:
                return {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5],
                    'subsample': [0.8, 0.9, 1.0]
                }
            elif 'XGBRegressor' in model_name:
                return {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.1, 1],
                    'reg_lambda': [0, 0.1, 1]
                }
            elif 'LGBMRegressor' in model_name:
                return {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'num_leaves': [15, 31, 63],
                    'max_depth': [-1, 5, 10],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            elif 'SVR' in model_name:
                return {
                    'C': [0.1, 1, 10, 100],
                    'epsilon': [0.01, 0.1, 0.5],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto']
                }
            elif 'KNeighborsRegressor' in model_name:
                return {
                    'n_neighbors': [3, 5, 7, 11, 15],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]
                }

        else:
            if 'LogisticRegression' in model_name:
                return {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'l1_ratio': [0],
                    'solver': ['lbfgs', 'liblinear'],
                    'max_iter': [100, 500, 1000]
                }
            elif 'RandomForestClassifier' in model_name:
                return {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            elif 'GradientBoostingClassifier' in model_name:
                return {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5],
                    'subsample': [0.8, 0.9, 1.0]
                }
            elif 'XGBClassifier' in model_name:
                return {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.1, 1],
                    'reg_lambda': [0, 0.1, 1]
                }
            elif 'LGBMClassifier' in model_name:
                return {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'num_leaves': [15, 31, 63],
                    'max_depth': [-1, 5, 10],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            elif 'SVC' in model_name:
                return {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto'],
                    'probability': [True]
                }
            elif 'KNeighborsClassifier' in model_name:
                return {
                    'n_neighbors': [3, 5, 7, 11, 15],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]
                }
        return {}
        
    def _fit_and_search(self, model):
        param_grid = self._get_param_grid(model)
        
        if not param_grid or self.searcher is None or self.cv is None:
            print(f'model training {model.__class__.__name__} without search hyperparams')
            model.fit(self.X, self.y)
            return model, None, None
        
        print(f'find hyperparams for {model.__class__.__name__}...')
        
        if self.searcher == 'RandomizedSearchCV':
            searcher = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=min(self.n_iter, len(param_grid) * 10),
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=-1,
                random_state=self.random_state,
                verbose=0,
                return_train_score=False
            )
        elif self.searcher == 'GridSearchCV':
            searcher = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=-1,
                verbose=0,
                return_train_score=False
            )
        else:
            raise ValueError(f'unknown searcher: {self.searcher}')
        
        try:
            searcher.fit(self.X, self.y)
            return searcher.best_estimator_, searcher.best_params_, searcher.best_score_
        
        except Exception as e:
            print(f'error training {model.__class__.__name__}: {e}')    
            return model, None, None
                                            
    def run(self):    
        if self.X is None or self.y is None:
            raise ValueError('X and y must be provided before running')

        print(f'type: {self.type_model}')
        print(f'amount models: {len(self.model)}')
        print(f'searcher: {self.searcher}')
        print(f'cross-validation: {self.cv} folds')
        
        for i, model in enumerate(self.model):
            print(f'[{i+1}/{len(self.model)}] Handling {model.__class__.__name__}...')
            
            best_model, best_params, best_score = self._fit_and_search(model)
            
            self.best_models_.append(best_model)
            self.best_params_.append(best_params)
            self.best_scores_.append(best_score)
            
        
            if best_score is not None:
                print(f'\tScore: {best_score:.4f}')
            if best_params:
                print(f'\t Params: {best_params}')
            print()
            
        self.results_df_ = pd.DataFrame({
            'model' : [m.__class__.__name__ for m in self.best_models_],
            'score' : self.best_scores_,
            'params' : self.best_params_
        }).sort_values('score', ascending=False)
        
        print('results')
        print(self.results_df_.to_string(index=False))
        
    def get_results(self):
        if self.results_df_ is None:
            print('run() first')
            return None 
        
        return self.results_df_
    
    def _get_models_dict(self):
        return {
            'linear': LinearRegression() if self.type_model == 'regression' else LogisticRegression(max_iter=1000),
            'rf': RandomForestRegressor(random_state=self.random_state) if self.type_model == 'regression' else RandomForestClassifier(random_state=self.random_state),
            'gb': GradientBoostingRegressor(random_state=self.random_state) if self.type_model == 'regression' else GradientBoostingClassifier(random_state=self.random_state),
            'xgboost': XGBRegressor(random_state=self.random_state) if self.type_model == 'regression' else XGBClassifier(random_state=self.random_state),
            'lgbm': LGBMRegressor(verbose=-1, random_state=self.random_state) if self.type_model == 'regression' else LGBMClassifier(verbose=-1, random_state=self.random_state),
            'svm': SVR() if self.type_model == 'regression' else SVC(random_state=self.random_state),
            'knn': KNeighborsRegressor() if self.type_model == 'regression' else KNeighborsClassifier()
        }