import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf 

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, SGDRegressor
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

st.set_page_config(page_title="Blackbox AI", layout="wide")

st.title("Blackbox AI")

# Step 1: Upload CSV file
st.sidebar.header("Step 1: Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head(5))

    col1, col2 = st.columns(2)
    with col1:
        st.header("Correlation Heatmap")
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    with col2:

        st.header("Display Distributions")

        subcol1, subcol2 = st.columns(2)

        with subcol1:
            bins = st.slider("Select Number of Bins", min_value=5, max_value=100, step=5, value=50)

        with subcol2:
            selected_feature = st.selectbox("Select Feature to Display Distribution", df.select_dtypes(include='number').columns)
       
        if selected_feature:
            st.write(f"### Distribution of {selected_feature}")
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[selected_feature], kde=True, ax=ax, bins=bins, palette="mako")
            st.pyplot(fig)

    # Step 2: Select features and target
    st.sidebar.header("Step 2: Select Features and Target")
    features = st.sidebar.multiselect("Select Features", df.columns)
    target = st.sidebar.selectbox("Select Target", set(df.columns) - set(features))

    # Step 3: Select classification or regression
    st.sidebar.header("Step 3: Select Task Type")
    task_type = st.sidebar.radio("Task Type", ("Regression", "Classification"))

    # Step 4: Select preprocessing steps
    st.sidebar.header("Step 4: Select Preprocessing Steps")
    scaler = st.sidebar.selectbox("Select Scaler", ["StandardScaler", "MinMaxScaler", "None"])
    encoder = st.sidebar.selectbox("Select Encoder", ["OneHotEncoder", "OrdinalEncoder", "None"])
    nullHandlerNumeric = st.sidebar.selectbox("Select Null Handler for Numeric Data", ["KNNImputer", "Mean", "Median", "MostFrequent", "None"])
    nullHandlerCategorical = st.sidebar.selectbox("Select Null Handler for Categorical Data", ["KNNImputer", "MostFrequent", "None"])
    custom_model = CustomModel(df, task_type, features, target, scaler, encoder, nullHandlerNumeric, nullHandlerCategorical)
    if st.sidebar.button("Preprocess Data"):
        st.session_state.df_preprocess = custom_model.preprocessData()

    if "df_preprocess" in st.session_state:
        st.write("### Preprocessed Data")
        st.dataframe(st.session_state.df_preprocess.head())

    # Step 5: Select model type
    st.sidebar.header("Step 5: Select Model Type")
    if task_type == "Regression":
        model_type = st.sidebar.selectbox("Model Type", ("Linear", "SGDRegressor", "RandomForestRegressor", "NeuralNetwork", "XGBoostRegressor"))
    else:
        model_type = st.sidebar.selectbox("Model Type", ("Logistic", "RandomForestClassifier", "SVM", "DecisionTree", "KNN", "NeuralNetwork", "XGBoostClassifier"))
    custom_model.modelType = model_type

    # Step 6: Select metrics
    st.sidebar.header("Step 6: Select Metrics")
    if task_type == "Regression":
        metrics = st.sidebar.multiselect("Select Metrics", ["MSE", "RMSE", "MAE", "R2 Score"])
    else:
        metrics = st.sidebar.multiselect("Select Metrics", ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "Confusion Matrix"])

    # Step 7: Train model
    if st.sidebar.button("Train Model"):
        st.write("### Training Model...")
        # Create and train the model using CustomModel class
        best_params = custom_model.trainModel()

        st.write("### Model trained successfully!")
        st.write("**Best Parameters:**")
        st.write(pd.DataFrame([best_params]))

        # Evaluate the model
        metrics_dict = custom_model.evaluateModel(metrics)
        st.write("### Model Evaluation")
        st.write(metrics_dict)




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
