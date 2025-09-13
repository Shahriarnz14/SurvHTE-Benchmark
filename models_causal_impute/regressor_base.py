from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

class RegressorBaseLearner:
    """
    Wrapper around common regression models with built-in hyperparameter tuning using cross-validation.
    
    Supported models:
    - 'ridge': Ridge regression
    - 'lasso': Lasso regression
    - 'rf': Random Forest Regressor
    - 'gbr': Gradient Boosting Regressor
    - 'xgb': XGBoost Regressor
    """
    def __init__(self, model_name='ridge', cv=5, scoring='neg_mean_squared_error', random_state=42):
        """
        Initialize the regressor with cross-validation.
        
        Parameters:
        - model_name (str): Name of the regression model.
        - cv (int): Number of cross-validation folds.
        - scoring (str): Scoring metric for GridSearchCV.
        - random_state (int): Random seed for reproducibility.
        """
        self.model_name = model_name
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.model = None
        self._init_model_and_grid()

    def _init_model_and_grid(self):
        """Initialize model instance and its hyperparameter grid."""
        if self.model_name == 'ridge':
            model = Ridge()
            param_grid = {'model__alpha': [0.01, 0.1, 1, 10, 100]}
        elif self.model_name == 'lasso':
            model = Lasso(max_iter=5000)
            param_grid = {'model__alpha': [0.001, 0.01, 0.1, 1, 10]}
        elif self.model_name == 'rf':
            model = RandomForestRegressor(random_state=self.random_state)
            param_grid = {
                'model__n_estimators': [50, 100],
                'model__max_depth': [3, 5, None]
            }
        elif self.model_name == 'gbr':
            model = GradientBoostingRegressor(random_state=self.random_state)
            param_grid = {
                'model__n_estimators': [50, 100],
                'model__learning_rate': [0.01, 0.1],
                'model__max_depth': [3, 5]
            }
        elif self.model_name == 'xgb':
            model = XGBRegressor(random_state=self.random_state, verbosity=0, n_jobs=-1)
            param_grid = {
                'model__n_estimators': [50, 100],
                'model__learning_rate': [0.01, 0.1],
                'model__max_depth': [3, 5]
            }
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        self.grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1
        )

    def fit(self, X, y):
        """
        Fit the model with hyperparameter tuning on the training data.
        
        Parameters:
        - X (np.ndarray): Training features.
        - y (np.ndarray): Training targets.
        """
        self.grid_search.fit(X, y)
        self.model = self.grid_search.best_estimator_

    def predict(self, X):
        """
        Predict on new data using the best-fitted model.
        
        Parameters:
        - X (np.ndarray): Test features.
        
        Returns:
        - np.ndarray: Predicted values.
        """
        if self.model is None:
            raise RuntimeError("Model not fit yet.")
        return self.model.predict(X)
    
    def evaluate(self, X_test, Y_test):
        """
        Evaluate model performance on test data.
        
        Parameters:
        - X_test (np.ndarray): Test features
        - Y_test (np.ndarray): Test targets (time)
        
        Returns:
        - dict or float: Regression Performance metrics
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        y_pred = self.model.predict(X_test)
        
        mae = mean_absolute_error(Y_test, y_pred)
        r2 = r2_score(Y_test, y_pred)
        
        return {'mae': mae, 'r2': r2}

    def get_best_params(self):
        """Return the best hyperparameters found during training."""
        return self.grid_search.best_params_
