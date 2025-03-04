import streamlit as st
import pandas as pd
from models import CustomModel
import seaborn as sns
import matplotlib.pyplot as plt


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
        corr = df.corr()
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