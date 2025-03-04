import time 
import sys 

import pandas as pd 
import numpy as np 

import tensorflow as tf 

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 

from xgboost import XGBClassifier, XGBRegressor

class CustomModel:
    
    def __init__(self, df, taskType, x_features, y_feature, scalerType, encoderType, nullHandlerNumeric, nullHandlerCategorical, modelType=None):
        self.df = df
        self.taskType = taskType
        self.modelType = modelType
        self.X = self.df[x_features]
        self.y = self.df[y_feature]

        match scalerType:
            case 'StandardScaler':
                self.scalerType = StandardScaler()
            case 'MinMaxScaler':
                self.scalerType = MinMaxScaler()
            case 'None':
                self.scalerType = None
        
        match encoderType:
            case 'OneHotEncoder':
                self.encoderType = OneHotEncoder(sparse_output=False)
            case 'OrdinalEncoder':
                self.encoderType = OrdinalEncoder()
            case 'None':
                self.encoderType = None

        match nullHandlerNumeric:
            case 'KNNImputer':
                self.nullHandlerNumeric = KNNImputer(n_neighbors=5)
            case 'Mean':
                self.nullHandlerNumeric = SimpleImputer(strategy='mean')
            case 'Median':
                self.nullHandlerNumeric = SimpleImputer(strategy='median')
            case 'MostFrequent':
                self.nullHandlerNumeric = SimpleImputer(strategy='most_frequent')
            case 'None':
                self.nullHandlerNumeric = None

        match nullHandlerCategorical:
            case 'KNNImputer':
                self.nullHandlerCategorical = KNNImputer(n_neighbors=5)
            case 'MostFrequent':
                self.nullHandlerCategorical = SimpleImputer(strategy='most_frequent')
            case 'None':
                self.nullHandlerCategorical = None

    def generatePreprocessor(self):
        numerical_pipeline = Pipeline([('Imputer', self.nullHandlerNumeric),
                               ('Scaler', self.scalerType)])

        categorical_pipeline = Pipeline([('Encoder', self.encoderType),
                                 ('Imputer', self.nullHandlerCategorical)])

        data_pipeline = ColumnTransformer([('numerical', numerical_pipeline, self.X.select_dtypes(exclude='object').columns),
                                   ('categorical', categorical_pipeline, self.X.select_dtypes(include='object').columns)])

        return data_pipeline
    
    def generateModel(self):
        if self.modelType == 'Linear':
            model = LinearRegression()
            param_grid = {'fit_intercept': [True]}

        elif self.modelType == 'SGDRegressor':
            model = SGDRegressor()
            param_grid = {
                'loss': ['squared_loss', 'huber'],
                'penalty': ['l2', 'elasticnet'],
                'alpha': [1e-4, 1e-3],
                # For elasticnet, a single l1_ratio value can suffice for initial tuning
                'l1_ratio': [0.15],
                'learning_rate': ['optimal'],
            }

        elif self.modelType == 'Logistic':
            model = LogisticRegression()
            c_space = np.logspace(-5, 8, 15)
            param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

        elif self.modelType == 'RandomForestRegressor':
            model = RandomForestRegressor()
            param_grid = {
                'n_estimators': [50, 100],
                'max_features': ['sqrt', 'log2', None],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'bootstrap': [True]
            }
        elif self.modelType == 'RandomForestClassifier':
            model = RandomForestClassifier()
            param_grid = {
                'n_estimators': [50, 100],
                'max_features': ['sqrt', 'log2', None],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'bootstrap': [True]
            }
            
        elif self.modelType == 'SVM':
            model = SVR()
            param_grid = {'C': [0.1, 1, 10, 100, 1000],
                          'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                          'kernel': ['rbf']}

        elif self.modelType == 'DecisionTree':
            model = DecisionTreeClassifier()
            param_grid = {'criterion': ['gini', 'entropy'],
                          'splitter': ['best', 'random'],
                          'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                          'min_samples_split': [2, 5, 10],
                          'min_samples_leaf': [1, 2, 4]}

        elif self.modelType == 'KNN':
            model = KNeighborsClassifier()
            param_grid = {'n_neighbors': np.arange(1, 25)}

        elif self.modelType =='NeuralNetwork':
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.InputLayer(shape=(self.X_train_preprocessed.shape[1],)))
            model.add(tf.keras.layers.Dense(32, activation='relu'))
            model.add(tf.keras.layers.Dense(64, activation='relu'))
            model.add(tf.keras.layers.Dense(32, activation='relu'))
            model.add(tf.keras.layers.Dense(1, activation='linear'))
            model.compile(optimizer='adam', loss='mean_squared_error')


        elif self.modelType == 'XGBoostRegressor':
            model = XGBRegressor()
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }

        elif self.modelType == 'XGBoostClassifier':
            model = XGBClassifier()
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }

        return GridSearchCV(model, param_grid, cv=5)
    
    def preprocessData(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)
        self.preprocessor = self.generatePreprocessor()
        self.X_train_preprocessed = self.preprocessor.fit_transform(self.X_train)
        self.X_test_preprocessed = self.preprocessor.transform(self.X_test)
        return pd.DataFrame(self.X_train_preprocessed, columns=self.preprocessor.get_feature_names_out(self.X_train.columns))
    
    def trainModel(self):
        self.preprocessData()
        self.model = self.generateModel()
        self.model.fit(self.X_train_preprocessed, self.y_train)
        return self.model.best_params_
    
    def evaluateModel(self, metrics):
        y_pred = self.model.predict(self.X_test_preprocessed)
        metrics_dict = {}
        
        if 'Accuracy' in metrics:
            metrics_dict['Accuracy'] = accuracy_score(self.y_test, y_pred)
        
        if 'Precision' in metrics:
            metrics_dict['Precision'] = precision_score(self.y_test, y_pred)
        
        if 'Recall' in metrics:
            metrics_dict['Recall'] = recall_score(self.y_test, y_pred)
        
        if 'F1 Score' in metrics:
            metrics_dict['F1 Score'] = f1_score(self.y_test, y_pred)
        
        if 'MSE' in metrics:
            metrics_dict['MSE'] = mean_squared_error(self.y_test, y_pred)
        
        if 'RMSE' in metrics:
            metrics_dict['RMSE'] = root_mean_squared_error(self.y_test, y_pred)
        
        if 'MAE' in metrics:
            metrics_dict['MAE'] = mean_absolute_error(self.y_test, y_pred)
        
        if 'R2 Score' in metrics:
            metrics_dict['R2 Score'] = r2_score(self.y_test, y_pred)

        if 'ROC AUC' in metrics:
            metrics_dict['ROC AUC'] = roc_auc_score(self.y_test, y_pred)
        
        if 'Confusion Matrix' in metrics:
            metrics_dict['Confusion Matrix'] = confusion_matrix(self.y_test, y_pred)
        
        return metrics_dict
