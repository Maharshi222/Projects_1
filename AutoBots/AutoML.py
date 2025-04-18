#Currently in Progress

import pandas as pd
import numpy as np
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Tuple
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import get_scorer
import logging
import traceback
from datetime import datetime
import json
import hashlib
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
import warnings
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class SystemState(Enum):
    IDLE = auto()
    DATA_LOADING = auto()
    DATA_ANALYSIS = auto()
    PREPROCESSING = auto()
    FEATURE_ENGINEERING = auto()
    MODEL_SELECTION = auto()
    TRAINING = auto()
    EVALUATION = auto()
    ERROR_RESOLUTION = auto()
    COMPLETED = auto()
    FAILED = auto()

class TaskType(Enum):
    CLASSIFICATION = auto()
    REGRESSION = auto()
    CLUSTERING = auto()
    TIMESERIES = auto()

class AutoMLConfig(BaseModel):
    target_column: Optional[str] = None
    task_type: Optional[TaskType] = None
    time_limit: int = 600  # seconds
    metric: Optional[str] = None
    preprocessing_steps: Dict[str, Any] = {}
    allowed_models: List[str] = []
    random_state: int = 42
    test_size: float = 0.2
    verbose: bool = True
    max_features: int = 20  # Maximum number of features to keep
    feature_selection: bool = True  # Whether to perform feature selection

class ModelResult(BaseModel):
    model_name: str
    parameters: Dict[str, Any]
    train_score: float
    test_score: float
    training_time: float
    model_hash: str
    feature_importances: Optional[Dict[str, float]] = None

class DataAnalysisResult(BaseModel):
    suggested_target: str
    suggested_task_type: TaskType
    missing_values: Dict[str, float]
    feature_types: Dict[str, str]
    correlations: Dict[str, float]
    class_distribution: Optional[Dict[str, float]] = None

class ErrorResolutionResult(BaseModel):
    success: bool
    retry_state: SystemState
    solution: Optional[str] = None

class DataLoaderAgent:
    def load(self, data_source: str) -> pd.DataFrame:
        """Load data from various sources with improved error handling"""
        try:
            if data_source.endswith('.csv'):
                return pd.read_csv(data_source, low_memory=False)
            elif data_source.endswith(('.xlsx', '.xls')):
                return pd.read_excel(data_source)
            elif data_source.startswith('http'):
                return pd.read_csv(data_source)
            else:
                raise ValueError(f"Unsupported data source: {data_source}")
        except Exception as e:
            logger.error(f"Failed to load data from {data_source}: {str(e)}")
            raise

class DataAnalyzerAgent:
    def analyze(self, data: pd.DataFrame) -> DataAnalysisResult:
        """Perform comprehensive data analysis with more robust checks"""
        if data.empty:
            raise ValueError("Empty dataframe provided for analysis")
            
        missing_values = (data.isnull().mean() * 100).to_dict()
        
        feature_types = {}
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                feature_types[col] = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(data[col]):
                feature_types[col] = "datetime"
            else:
                feature_types[col] = "categorical"
        
        suggested_target = self._suggest_target(data, feature_types)
        print(f"suggested target the model found is: {suggested_target}")
        suggested_task_type = self._suggest_task_type(data, suggested_target)
        print(f"suggested task type the model found is:{suggested_task_type}")
        
        correlations = {}
        if suggested_target and feature_types.get(suggested_target) == "numeric":
            numeric_data = data.select_dtypes(include='number')
            if suggested_target in numeric_data.columns:
                correlations = numeric_data.corr()[suggested_target].to_dict()
        
        class_distribution = None
        if suggested_target and suggested_task_type == TaskType.CLASSIFICATION:
            class_distribution = (data[suggested_target].value_counts(normalize=True) * 100).to_dict()
        
        return DataAnalysisResult(
            suggested_target=suggested_target,
            suggested_task_type=suggested_task_type,
            missing_values=missing_values,
            feature_types=feature_types,
            correlations=correlations,
            class_distribution=class_distribution
        )
    
    def _suggest_target(self, data: pd.DataFrame, feature_types: Dict[str, str]) -> Optional[str]:
        target_candidates = ['target', 'label', 'class', 'output', 'amount', 'quantity', 
                           'price', 'value', 'score', 'rating']
        
        for candidate in target_candidates:
            for col in data.columns:
                if candidate.lower() in col.lower():
                    return col
                    
        numeric_cols = [col for col, typ in feature_types.items() if typ == "numeric"]
        if numeric_cols:
            print(data[numeric_cols])
            return data[numeric_cols].var().idxmax()
            
        return data.columns[-1] if len(data.columns) > 0 else None
    
    def _suggest_task_type(self, data: pd.DataFrame, target_column: Optional[str]) -> TaskType:
        if not target_column:
            return TaskType.CLUSTERING
            
        unique_values = data[target_column].nunique()
        
        if pd.api.types.is_numeric_dtype(data[target_column]):
            return TaskType.REGRESSION if unique_values > 10 else TaskType.CLASSIFICATION
        else:
            return TaskType.CLASSIFICATION if unique_values < 20 else TaskType.CLUSTERING

class PreprocessingAgent:
    def process(self, data: pd.DataFrame, config: AutoMLConfig) -> Tuple[pd.DataFrame, Any]:
        """Handle all data preprocessing with proper train/test separation"""
        # Split data first to avoid leakage
        if config.target_column not in data.columns:
            raise ValueError(f"Target column '{config.target_column}' not found in data")
            
        X = data.drop(columns=[config.target_column])
        y = data[config.target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state
        )
        
        # Identify numeric and categorical columns
        numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
        cat_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

        # 🔥 Ensure uniform data types in categorical columns (avoid int/str mix)
        for col in cat_cols:
            X_train[col] = X_train[col].astype(str)
            X_test[col] = X_test[col].astype(str)

        
        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, cat_cols)
            ])
        
        # Fit and transform the training data
        X_train_processed = preprocessor.fit_transform(X_train)
        
        # Transform the test data
        X_test_processed = preprocessor.transform(X_test)
        
        # Get feature names after preprocessing
        feature_names = numeric_cols.copy()
        if cat_cols:
            ohe = preprocessor.named_transformers_['cat'].named_steps['encoder']
            cat_features = ohe.get_feature_names_out(cat_cols)
            feature_names.extend(cat_features)
        
        # Convert back to DataFrame
        #X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names)
        X_train_processed = pd.DataFrame.sparse.from_spmatrix(X_train_processed, columns=feature_names)

        X_test_processed = pd.DataFrame.sparse.from_spmatrix(X_test_processed, columns=feature_names)
        
        return X_train_processed, X_test_processed, y_train, y_test, preprocessor

class FeatureEngineeringAgent:
    def engineer(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                config: AutoMLConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Perform feature engineering with proper train/test separation"""
        # Only create interaction features for numeric columns
        numeric_cols = X_train.select_dtypes(include=np.number).columns
        
        # Limit the number of interaction features to avoid explosion
        if len(numeric_cols) > 1:
            for i, col1 in enumerate(numeric_cols[:5]):  # Limit to first 5 columns
                for col2 in numeric_cols[i+1:i+3]:  # Limit to next 2 columns
                    if f"{col1}_x_{col2}" not in X_train.columns:
                        X_train[f"{col1}_x_{col2}"] = X_train[col1] * X_train[col2]
                        X_test[f"{col1}_x_{col2}"] = X_test[col1] * X_test[col2]
        
        # Feature selection if enabled
        if config.feature_selection and len(X_train.columns) > config.max_features:
            X_train, X_test = self._select_features(X_train, X_test, config)
            
        return X_train, X_test
    
    def _select_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                        config: AutoMLConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Select top features based on statistical tests"""
        if config.task_type == TaskType.CLASSIFICATION:
            selector = SelectKBest(f_classif, k=min(config.max_features, X_train.shape[1]))
        else:
            selector = SelectKBest(f_regression, k=min(config.max_features, X_train.shape[1]))
            
        selector.fit(X_train, config.target_column)
        selected_cols = X_train.columns[selector.get_support()]
        
        return X_train[selected_cols], X_test[selected_cols]

class ModelSelectionAgent:
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.models = []
        self.best_model = None
        
    def select_and_train(self, X_train, y_train, X_test, y_test) -> List[ModelResult]:
        model_configs = self._get_model_configs()
        
        for model_name, model_class, params in model_configs:
            try:
                # Skip if model is not in allowed_models (when specified)
                if self.config.allowed_models and model_name.lower() not in [m.lower() for m in self.config.allowed_models]:
                    continue
                    
                model = model_class(**params)
                start_time = datetime.now()
                model.fit(X_train, y_train)
                train_time = (datetime.now() - start_time).total_seconds()
                
                # Get appropriate scorer
                scorer = self._get_scorer()
                train_score = scorer(model, X_train, y_train)
                test_score = scorer(model, X_test, y_test)
                
                # Get feature importances if available
                feature_importances = self._get_feature_importances(model, X_train.columns)
                
                model_hash = hashlib.md5(
                    f"{model_name}{params}{train_score}".encode()
                ).hexdigest()
                
                result = ModelResult(
                    model_name=model_name,
                    parameters=params,
                    train_score=train_score,
                    test_score=test_score,
                    training_time=train_time,
                    model_hash=model_hash,
                    feature_importances=feature_importances
                )
                self.models.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to train {model_name}: {str(e)}")
                continue
        
        return self.models
    
    def _get_scorer(self):
        """Get appropriate scorer based on task type and config"""
        if self.config.metric:
            return get_scorer(self.config.metric)
            
        if self.config.task_type == TaskType.CLASSIFICATION:
            return get_scorer('accuracy')
        elif self.config.task_type == TaskType.REGRESSION:
            return get_scorer('r2')
        else:
            return get_scorer('accuracy')
    
    def _get_feature_importances(self, model, feature_names):
        """Extract feature importances in a consistent way"""
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            if len(model.coef_.shape) == 1:
                return dict(zip(feature_names, model.coef_))
            else:
                # For multi-class classification
                return {f"{col}_class_{i}": coef 
                       for i, coefs in enumerate(model.coef_) 
                       for col, coef in zip(feature_names, coefs)}
        else:
            # Calculate permutation importance as fallback
            try:
                result = permutation_importance(
                    model, X_train, y_train, n_repeats=5, random_state=self.config.random_state
                )
                return dict(zip(feature_names, result.importances_mean))
            except:
                return None
    
    def get_best_model(self) -> Optional[ModelResult]:
        if not self.models:
            return None
            
        if self.config.task_type == TaskType.CLASSIFICATION:
            return max(self.models, key=lambda x: x.test_score)
        else:
            return min(self.models, key=lambda x: x.test_score)
    
    def _get_model_configs(self) -> List[Tuple[str, Any, Dict[str, Any]]]:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
        from sklearn.svm import SVC, SVR
        from xgboost import XGBClassifier, XGBRegressor
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        from lightgbm import LGBMClassifier, LGBMRegressor
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        
        if self.config.task_type == TaskType.CLASSIFICATION:
            return [
                ("Random Forest", RandomForestClassifier, 
                 {'n_estimators': 30, 'random_state': self.config.random_state}),
                ("XGBoost", XGBClassifier, 
                 {'random_state': self.config.random_state, 'eval_metric': 'logloss'}),
                ("Logistic Regression", LogisticRegression, 
                 {'max_iter': 1000, 'random_state': self.config.random_state}),
                # ("SVM", SVC, 
                #  {'probability': True, 'random_state': self.config.random_state}),
                ("Gradient Boosting", GradientBoostingClassifier, 
                 {'random_state': self.config.random_state}),
                ("LightGBM", LGBMClassifier, 
                 {'random_state': self.config.random_state}),
                ("KNN", KNeighborsClassifier, {})
            ]
        elif self.config.task_type == TaskType.REGRESSION:
            return [
                ("Random Forest", RandomForestRegressor, 
                 {'n_estimators': 30, 'random_state': self.config.random_state}),
                ("XGBoost", XGBRegressor, 
                 {'random_state': self.config.random_state}),
                ("Linear Regression", LinearRegression, {}),
                ("Ridge Regression", Ridge, 
                 {'random_state': self.config.random_state}),
                ("Lasso Regression", Lasso, 
                 {'random_state': self.config.random_state}),
                ("SVR", SVR, {}),
                ("Gradient Boosting", GradientBoostingRegressor, 
                 {'random_state': self.config.random_state}),
                ("LightGBM", LGBMRegressor, 
                 {'random_state': self.config.random_state}),
                ("KNN", KNeighborsRegressor, {})
            ]
        else:
            return []

class EvaluationAgent:
    def evaluate(self, model_result: ModelResult, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Perform comprehensive model evaluation"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
            mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
        )
        
        evaluation = {
            "model_name": model_result.model_name,
            "test_score": model_result.test_score
        }
        
        if model_result.feature_importances:
            top_features = dict(sorted(
                model_result.feature_importances.items(),
                key=lambda item: abs(item[1]), 
                reverse=True
                )[:5])
            evaluation["top_features"] = top_features
        
        return evaluation

class ErrorResolverAgent:
    def resolve(self, error: Exception, context: Any) -> ErrorResolutionResult:
        error_type = type(error).__name__
        error_msg = str(error)
        
        solutions = {
            "ValueError": self._handle_value_error,
            "KeyError": self._handle_key_error,
            "MemoryError": self._handle_memory_error,
            "ImportError": self._handle_import_error,
            "ConvergenceWarning": self._handle_convergence_error
        }
        
        handler = solutions.get(error_type, self._handle_generic_error)
        return handler(error, context)
    
    def _handle_value_error(self, error: Exception, context: Any) -> ErrorResolutionResult:
        error_msg = str(error)
        
        if "could not convert string to float" in error_msg:
            return ErrorResolutionResult(
                success=True,
                retry_state=SystemState.PREPROCESSING,
                solution="String to float conversion error - retrying with better preprocessing"
            )
        elif "Found unknown categories" in error_msg:
            return ErrorResolutionResult(
                success=True,
                retry_state=SystemState.PREPROCESSING,
                solution="Unknown categories found - retrying with more robust encoding"
            )
        
        return ErrorResolutionResult(
            success=False,
            retry_state=SystemState.FAILED
        )
    
    def _handle_key_error(self, error: Exception, context: Any) -> ErrorResolutionResult:
        return ErrorResolutionResult(
            success=True,
            retry_state=SystemState.DATA_ANALYSIS,
            solution="Column not found - re-analyzing data structure"
        )
    
    def _handle_memory_error(self, error: Exception, context: Any) -> ErrorResolutionResult:
        return ErrorResolutionResult(
            success=True,
            retry_state=SystemState.PREPROCESSING,
            solution="Memory error - retrying with reduced data size and features"
        )
    
    def _handle_import_error(self, error: Exception, context: Any) -> ErrorResolutionResult:
        missing_pkg = str(error).split(" ")[-1]
        return ErrorResolutionResult(
            success=False,
            retry_state=SystemState.FAILED,
            solution=f"Required package {missing_pkg} not installed"
        )
    
    def _handle_convergence_error(self, error: Exception, context: Any) -> ErrorResolutionResult:
        return ErrorResolutionResult(
            success=True,
            retry_state=SystemState.MODEL_SELECTION,
            solution="Model failed to converge - trying different model or parameters"
        )
    
    def _handle_generic_error(self, error: Exception, context: Any) -> ErrorResolutionResult:
        return ErrorResolutionResult(
            success=False,
            retry_state=SystemState.FAILED,
            solution=f"Unexpected error: {str(error)}"
        )

class AutoMLSystem:
    def __init__(self, config: AutoMLConfig):
        self.state = SystemState.IDLE
        self.config = config
        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.models: List[ModelResult] = []
        self.best_model: Optional[ModelResult] = None
        self.errors = []
        self.start_time = None
        self.current_operation = None
        self.preprocessor = None
        self._initialize_agents()
    def visualize_results(self):
        """Visualize all relevant results based on the task type"""
        visualizer = AutoMLVisualizer(self)
        
        # Data exploration visualizations
        visualizer.visualize_data_exploration()
        
        # Model-specific visualizations
        if self.best_model:
            from sklearn.base import clone
            model = clone(self.best_model)
            model.fit(self.X_train, self.y_train)  # Re-fit for visualization
            
            if self.config.task_type == TaskType.CLASSIFICATION:
                visualizer.visualize_classification(model, self.X_test, self.y_test)
            elif self.config.task_type == TaskType.REGRESSION:
                visualizer.visualize_regression(model, self.X_test, self.y_test)
            elif self.config.task_type == TaskType.CLUSTERING:
                visualizer.visualize_clustering(model, self.X_test)
        
        # Model comparison visualization
        visualizer.visualize_model_performance()
        
        # Feature importance visualization
        if self.best_model:
            visualizer.visualize_feature_importance(self.best_model)

        
    def _initialize_agents(self):
        self.data_loader = DataLoaderAgent()
        self.analyzer = DataAnalyzerAgent()
        self.preprocessor_agent = PreprocessingAgent()
        self.feature_engineer = FeatureEngineeringAgent()
        self.model_selector = ModelSelectionAgent(self.config)
        self.error_resolver = ErrorResolverAgent()
        self.evaluator = EvaluationAgent()
        
    def run(self, data_source: str):
        self.start_time = datetime.now()
        try:
            self._load_data(data_source)
            self._analyze_data()
            self._preprocess_data()
            self._engineer_features()
            self._train_models()
            self._evaluate_models()
            self.state = SystemState.COMPLETED
        except Exception as e:
            self._handle_error(e)
        finally:
            self._cleanup()
            
    def _load_data(self, data_source: str):
        self._set_state(SystemState.DATA_LOADING, "Loading data")
        self.data = self.data_loader.load(data_source)
        
    def _analyze_data(self):
        self._set_state(SystemState.DATA_ANALYSIS, "Analyzing data")
        analysis_result = self.analyzer.analyze(self.data)
        
        if not self.config.target_column:
            self.config.target_column = analysis_result.suggested_target
            logger.info(f"Auto-detected target column: {self.config.target_column}")
            
        if not self.config.task_type:
            self.config.task_type = analysis_result.suggested_task_type
            logger.info(f"Auto-detected task type: {self.config.task_type.name}")
            
        # Log class distribution for classification tasks
        if (self.config.task_type == TaskType.CLASSIFICATION and 
            analysis_result.class_distribution):
            logger.info(f"Class distribution: {analysis_result.class_distribution}")
            
    def _preprocess_data(self):
        self._set_state(SystemState.PREPROCESSING, "Preprocessing data")
        (self.X_train, self.X_test, 
         self.y_train, self.y_test, 
         self.preprocessor) = self.preprocessor_agent.process(self.data, self.config)
        
    def _engineer_features(self):
        self._set_state(SystemState.FEATURE_ENGINEERING, "Engineering features")
        self.X_train, self.X_test = self.feature_engineer.engineer(
            self.X_train, self.X_test, self.config
        )
        
    def _train_models(self):
        self._set_state(SystemState.TRAINING, "Training models")
        self.models = self.model_selector.select_and_train(
            self.X_train, self.y_train, self.X_test, self.y_test
        )
        self.best_model = self.model_selector.get_best_model()
        if self.best_model:
            logger.info(f"Best model: {self.best_model.model_name} with test score: {self.best_model.test_score:.4f}")
        
    def _evaluate_models(self):
        self._set_state(SystemState.EVALUATION, "Evaluating models")
        self.evaluation_results = []
        for model_result in self.models:
            eval_result = self.evaluator.evaluate(model_result, self.X_test, self.y_test)
            self.evaluation_results.append(eval_result)
        
    def _handle_error(self, error: Exception):
        self.state = SystemState.ERROR_RESOLUTION
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "operation": self.current_operation,
            "error": str(error),
            "traceback": traceback.format_exc()
        }
        self.errors.append(error_info)
        logger.error(f"Error during {self.current_operation}: {error}")
        
        resolution = self.error_resolver.resolve(error, self)
        if resolution.success:
            logger.info(f"Error resolved automatically: {resolution.solution}")
            self.state = resolution.retry_state
            self._retry_operation()
        else:
            logger.error(f"Failed to resolve error automatically: {resolution.solution if resolution.solution else 'Unknown error'}")
            self.state = SystemState.FAILED
            
    def _retry_operation(self):
        retry_operations = {
            SystemState.DATA_LOADING: self._load_data,
            SystemState.DATA_ANALYSIS: self._analyze_data,
            SystemState.PREPROCESSING: self._preprocess_data,
            SystemState.FEATURE_ENGINEERING: self._engineer_features,
            SystemState.TRAINING: self._train_models,
            SystemState.EVALUATION: self._evaluate_models
        }
        
        if self.state in retry_operations:
            retry_operations[self.state]()
            
    def _set_state(self, state: SystemState, operation: str):
        self.state = state
        self.current_operation = operation
        logger.info(f"State changed to {state.name}: {operation}")
        
    def _cleanup(self):
        runtime = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"AutoML completed in {runtime:.2f} seconds. Final state: {self.state.name}")
        
    def get_results(self) -> Dict[str, Any]:
        return {
            "state": self.state.name,
            "best_model": self.best_model.model_dump() if self.best_model else None,
            "all_models": [m.model_dump() for m in self.models],
            "evaluations": self.evaluation_results if hasattr(self, 'evaluation_results') else [],
            "errors": self.errors,
            "config": self.config.model_dump()
        }

# Add these imports at the top of your file
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (confusion_matrix, classification_report, 
                            roc_curve, auc, precision_recall_curve)
from yellowbrick.classifier import ROCAUC, ClassificationReport, ConfusionMatrix
from yellowbrick.regressor import PredictionError, ResidualsPlot
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from yellowbrick.model_selection import FeatureImportances

import os
from datetime import datetime

class AutoMLVisualizer:
    def __init__(self, automl_system, output_dir="visualizations"):
        self.automl = automl_system
        self.config = automl_system.config
        self.console = Console()
        
        # Create output directory if it doesn't exist
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _save_figure(self, filename_prefix):
        """Helper method to save the current figure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        self.console.print(f"[green]Saved visualization to {filepath}[/green]")
        return filepath
        
    def visualize_data_exploration(self):
        """Visualize data distributions and relationships"""
        if self.automl.data is None:
            self.console.print("[red]No data available for visualization[/red]")
            return
            
        data = self.automl.data
        target = self.config.target_column
        
        # Create a figure with multiple subplots
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Target distribution
        plt.subplot(2, 2, 1)
        if target and target in data.columns:
            if self.config.task_type == TaskType.CLASSIFICATION:
                sns.countplot(x=target, data=data)
                plt.title(f'Distribution of {target}')
            else:
                sns.histplot(data[target], kde=True)
                plt.title(f'Distribution of {target}')
        else:
            plt.text(0.5, 0.5, 'No target column specified', 
                    ha='center', va='center')
            plt.title('Target Distribution')
        
        # Plot 2: Missing values heatmap
        plt.subplot(2, 2, 2)
        sns.heatmap(data.isnull(), cbar=False)
        plt.title('Missing Values Heatmap')
        
        # Plot 3: Numeric features correlation
        plt.subplot(2, 2, 3)
        numeric_cols = data.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 1:
            sns.heatmap(data[numeric_cols].corr(), annot=True, fmt='.2f')
            plt.title('Numeric Features Correlation')
        else:
            plt.text(0.5, 0.5, 'Not enough numeric features', 
                    ha='center', va='center')
            plt.title('Feature Correlation')
        
        # Plot 4: Pairplot for top 3 numeric features
        if len(numeric_cols) >= 3 and target and target in numeric_cols:
            plt.subplot(2, 2, 4)
            sns.pairplot(data[numeric_cols[:3].tolist() + [target]])
            plt.title('Pairplot of Top 3 Numeric Features')
        else:
            plt.text(0.5, 0.5, 'Not enough numeric features for pairplot', 
                    ha='center', va='center')
            plt.title('Feature Relationships')
        
        plt.tight_layout()
        return self._save_figure("data_exploration")
    
    def visualize_classification(self, model, X_test, y_test):
        """Visualize classification model performance"""
        if self.config.task_type != TaskType.CLASSIFICATION:
            self.console.print("[yellow]Not a classification task - skipping classification visualizations[/yellow]")
            return
            
        # Create a figure with multiple subplots
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Confusion Matrix
        plt.subplot(2, 2, 1)
        viz = ConfusionMatrix(model)
        viz.fit(X_test, y_test)
        viz.score(X_test, y_test)
        viz.finalize()
        
        # Plot 2: ROC Curve
        plt.subplot(2, 2, 2)
        viz = ROCAUC(model)
        viz.fit(X_test, y_test)
        viz.score(X_test, y_test)
        viz.finalize()
        
        # Plot 3: Classification Report
        plt.subplot(2, 2, 3)
        viz = ClassificationReport(model, support=True)
        viz.fit(X_test, y_test)
        viz.score(X_test, y_test)
        viz.finalize()
        
        # Plot 4: Feature Importance
        plt.subplot(2, 2, 4)
        try:
            viz = FeatureImportances(model, relative=False)
            viz.fit(X_test, y_test)
            viz.score(X_test, y_test)
            viz.finalize()
        except Exception as e:
            plt.text(0.5, 0.5, f'Feature importance not available:\n{str(e)}', 
                    ha='center', va='center')
            plt.title('Feature Importance')
        
        plt.tight_layout()
        return self._save_figure(f"classification_{model.__class__.__name__}")
    
    def visualize_regression(self, model, X_test, y_test):
        """Visualize regression model performance"""
        if self.config.task_type != TaskType.REGRESSION:
            self.console.print("[yellow]Not a regression task - skipping regression visualizations[/yellow]")
            return
            
        # Create a figure with multiple subplots
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Prediction Error
        plt.subplot(2, 2, 1)
        viz = PredictionError(model)
        viz.fit(X_test, y_test)
        viz.score(X_test, y_test)
        viz.finalize()
        
        # Plot 2: Residuals Plot
        plt.subplot(2, 2, 2)
        viz = ResidualsPlot(model)
        viz.fit(X_test, y_test)
        viz.score(X_test, y_test)
        viz.finalize()
        
        # Plot 3: Actual vs Predicted
        plt.subplot(2, 2, 3)
        y_pred = model.predict(X_test)
        sns.regplot(x=y_test, y=y_pred)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        
        # Plot 4: Feature Importance
        plt.subplot(2, 2, 4)
        try:
            viz = FeatureImportances(model, relative=False)
            viz.fit(X_test, y_test)
            viz.score(X_test, y_test)
            viz.finalize()
        except Exception as e:
            plt.text(0.5, 0.5, f'Feature importance not available:\n{str(e)}', 
                    ha='center', va='center')
            plt.title('Feature Importance')
        
        plt.tight_layout()
        return self._save_figure(f"regression_{model.__class__.__name__}")
    
    def visualize_clustering(self, model, X_test):
        """Visualize clustering results"""
        if self.config.task_type != TaskType.CLUSTERING:
            self.console.print("[yellow]Not a clustering task - skipping clustering visualizations[/yellow]")
            return
            
        # Create a figure with multiple subplots
        plt.figure(figsize=(15, 10))
        
        # Reduce dimensionality for visualization
        if X_test.shape[1] > 2:
            reducer = PCA(n_components=2)
            X_vis = reducer.fit_transform(X_test)
        else:
            X_vis = X_test.values
            
        # Plot 1: Cluster Visualization
        plt.subplot(2, 2, 1)
        if hasattr(model, 'labels_'):
            labels = model.labels_
        else:
            labels = model.predict(X_test)
            
        plt.scatter(X_vis[:, 0], X_vis[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.title('Cluster Visualization')
        
        # Plot 2: Elbow Method (if KMeans)
        plt.subplot(2, 2, 2)
        try:
            from sklearn.cluster import KMeans
            if isinstance(model, KMeans):
                visualizer = KElbowVisualizer(model, k=(2,10))
                visualizer.fit(X_test)
                visualizer.finalize()
            else:
                plt.text(0.5, 0.5, 'Elbow method only for KMeans', 
                        ha='center', va='center')
                plt.title('Elbow Method')
        except ImportError:
            plt.text(0.5, 0.5, 'sklearn not available', 
                    ha='center', va='center')
            plt.title('Elbow Method')
        
        # Plot 3: Silhouette Score
        plt.subplot(2, 2, 3)
        try:
            visualizer = SilhouetteVisualizer(model)
            visualizer.fit(X_test)
            visualizer.finalize()
        except Exception as e:
            plt.text(0.5, 0.5, f'Silhouette visualization failed:\n{str(e)}', 
                    ha='center', va='center')
            plt.title('Silhouette Score')
        
        # Plot 4: Feature Importance (if available)
        plt.subplot(2, 2, 4)
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                plt.bar(range(X_test.shape[1]), importances[indices])
                plt.xticks(range(X_test.shape[1]), X_test.columns[indices], rotation=90)
                plt.title('Feature Importances')
            else:
                plt.text(0.5, 0.5, 'No feature importances available', 
                        ha='center', va='center')
                plt.title('Feature Importance')
        except Exception as e:
            plt.text(0.5, 0.5, f'Feature importance failed:\n{str(e)}', 
                    ha='center', va='center')
            plt.title('Feature Importance')
        
        plt.tight_layout()
        return self._save_figure(f"clustering_{model.__class__.__name__}")
    
    def visualize_model_performance(self):
        """Visualize performance of all trained models"""
        if not self.automl.models:
            self.console.print("[red]No models available for visualization[/red]")
            return
            
        # Create comparison plot
        plt.figure(figsize=(10, 6))
        
        model_names = [m.model_name for m in self.automl.models]
        test_scores = [m.test_score for m in self.automl.models]
        
        sns.barplot(x=test_scores, y=model_names)
        plt.title('Model Comparison (Test Scores)')
        plt.xlabel('Test Score')
        plt.ylabel('Model')
        
        if self.config.task_type == TaskType.CLASSIFICATION:
            plt.xlim(0, 1)  # Classification scores typically between 0-1
        elif self.config.task_type == TaskType.REGRESSION:
            # For regression, we might have negative scores (like negative MSE)
            pass
            
        plt.tight_layout()
        return self._save_figure("model_comparison")
    
    def visualize_feature_importance(self, model=None):
        """Visualize feature importance for a specific model or the best model"""
        if model is None:
            if self.automl.best_model is None:
                self.console.print("[red]No best model available for feature importance visualization[/red]")
                return
            model = self.automl.best_model
            
        if not hasattr(model, 'feature_importances_') and not hasattr(model, 'coef_'):
            self.console.print("[yellow]Model doesn't support feature importance visualization[/yellow]")
            return
            
        plt.figure(figsize=(10, 6))
        
        try:
            viz = FeatureImportances(model, relative=False)
            if hasattr(self.automl, 'X_train'):
                viz.fit(self.automl.X_train, self.automl.y_train)
            else:
                self.console.print("[yellow]Training data not available for feature importance[/yellow]")
                return
            viz.finalize()
            plt.title('Feature Importance')
            plt.tight_layout()
            return self._save_figure(f"feature_importance_{model.__class__.__name__}")
        except Exception as e:
            self.console.print(f"[red]Failed to visualize feature importance: {str(e)}[/red]")
if __name__ == "__main__":
    console = Console()

    config = AutoMLConfig(
        target_column="Amount",  # Optional - can be auto-detected
        #task_type=TaskType,  # Optional - can be auto-detected
        time_limit=300,
        metric="neg_root_mean_squared_error",
        random_state=42,
        max_features=8,
        feature_selection=True
    )

    path = r"D:\maharshi bkup\D drive\AMUL_Sales_Analysis\Data\SalesInputData1.xlsx"
    output_dir = r"D:\maharshi bkup\D drive\AMUL_Sales_Analysis\generated_images"
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    automl = AutoMLSystem(config)

    console.rule("[bold cyan]🚀 AutoML Process Started")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:

        task = progress.add_task("📦 Loading data...", start=False)
        progress.start_task(task)
        automl._load_data(path)
        progress.update(task, description="✅ Data loaded")

        task = progress.add_task("🔎 Analyzing data...", start=False)
        progress.start_task(task)
        automl._analyze_data()
        progress.update(task, description="✅ Analysis done")

        task = progress.add_task("⚙️ Preprocessing...", start=False)
        progress.start_task(task)
        automl._preprocess_data()
        progress.update(task, description="✅ Preprocessing done")

        task = progress.add_task("🧠 Training model...", start=False)
        progress.start_task(task)
        automl._train_models()
        progress.update(task, description="✅ Model trained")

        task = progress.add_task("📊 Evaluating model...", start=False)
        progress.start_task(task)
        automl._evaluate_models()
        progress.update(task, description="✅ Evaluation done")

    console.rule("[bold green]✅ AutoML Pipeline Completed")

    # Get results
    results = automl.get_results()
    if results and results.get("best_model"):
        print(f"\nBest model: {results['best_model']['model_name']}")
        print(f"Test score: {results['best_model']['test_score']:.4f}")
        
        if results['best_model']['feature_importances']:
            print("\nTop 5 important features:")
            for feature, importance in sorted(
                results['best_model']['feature_importances'].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]:
                print(f"{feature}: {importance:.4f}")
    else:
        print("⚠️ No best model found. Check for earlier errors in model training or selection.")
    
    # Create visualizer with the specified output directory
    visualizer = AutoMLVisualizer(automl, output_dir=output_dir)
    
    # Generate and save all visualizations
    console.rule("[bold blue]📊 Generating Visualizations")
    
    # Data exploration visualizations
    data_viz_path = visualizer.visualize_data_exploration()
    console.print(f"📈 Data exploration saved to: [link={data_viz_path}]{data_viz_path}[/link]")
    
    # Model performance comparison
    model_comp_path = visualizer.visualize_model_performance()
    console.print(f"📊 Model comparison saved to: [link={model_comp_path}]{model_comp_path}[/link]")
    
    # Best model visualizations
    if results and results.get("best_model"):
        best_model = results['best_model']['model']
        X_test = automl.X_test if hasattr(automl, 'X_test') else None
        y_test = automl.y_test if hasattr(automl, 'y_test') else None
        
        if config.task_type == TaskType.CLASSIFICATION:
            model_viz_path = visualizer.visualize_classification(best_model, X_test, y_test)
            console.print(f"🎯 Classification results saved to: [link={model_viz_path}]{model_viz_path}[/link]")
        elif config.task_type == TaskType.REGRESSION:
            model_viz_path = visualizer.visualize_regression(best_model, X_test, y_test)
            console.print(f"📈 Regression results saved to: [link={model_viz_path}]{model_viz_path}[/link]")
        elif config.task_type == TaskType.CLUSTERING:
            model_viz_path = visualizer.visualize_clustering(best_model, X_test)
            console.print(f"🔮 Clustering results saved to: [link={model_viz_path}]{model_viz_path}[/link]")
        
        # Feature importance
        feat_imp_path = visualizer.visualize_feature_importance(best_model)
        if feat_imp_path:
            console.print(f"🔍 Feature importance saved to: [link={feat_imp_path}]{feat_imp_path}[/link]")
    
    console.rule("[bold green]✅ All Visualizations Saved")
