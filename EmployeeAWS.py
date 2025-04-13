import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc, r2_score, mean_squared_error,
    mean_absolute_error
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

st.set_page_config(layout="wide")
# Custom CSS for watermark and scoped buttons
watermark_css = """
<style>
    .watermark {
        position: fixed;
        bottom: 10px;
        right: 10px;
        font-size: 14px;
        color: gray;
        opacity: 0.7;
    }
    
    .custom-button > div.stButton > button {
        width: 100%;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
        color: #0047AB;
        text-align: center;
        background: white;
        border: 2px solid #0047AB;
        border-radius: 8px;
        cursor: pointer;
        transition: 0.3s;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }

    .custom-button > div.stButton > button:hover {
        background: #0047AB;
        color: white;
    }
</style>
<div class="watermark">Data Scientist ArunKaviJS</div>
"""
st.markdown(watermark_css, unsafe_allow_html=True)


st.markdown(
        """
        <style>
            div.stButton > button {
                width: 100%; /* Full Width Inside Column */
                height: 50px;
                font-size: 18px;
                font-weight: bold;
                color: #0047AB;  /* Ocean Blue Text */
                text-align: center;
                background: white; /* White Background */
                border: 2px solid #0047AB; /* Blue Border */
                border-radius: 8px;
                cursor: pointer;
                transition: 0.3s;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            }

            div.stButton > button:hover {
                background: #0047AB; /* Blue Background on Hover */
                color: white; /* White Text on Hover */
            }
        </style>
        """,
        unsafe_allow_html=True
    )

#for button backgroun
st.markdown(
        """
        <style>
            div.stButton > button {
                width: 100%; /* Full Width Inside Column */
                height: 50px;
                font-size: 18px;
                font-weight: bold;
                color: #0047AB;  /* Ocean Blue Text */
                text-align: center;
                background: white; /* White Background */
                border: 2px solid #0047AB; /* Blue Border */
                border-radius: 8px;
                cursor: pointer;
                transition: 0.3s;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            }

            div.stButton > button:hover {
                background: #0047AB; /* Blue Background on Hover */
                color: white; /* White Text on Hover */
            }
        </style>
        """,
        unsafe_allow_html=True
    )
# Initialize session state
def initialize_session_state():
    defaults = {
        "page": "Home",
        "df": None,
        "best_model": None,
        "best_model_name": "",
        "best_score": -np.inf,
        "target_type": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Navigation function
def navigate_to(page_name):
    st.session_state["page"] = page_name

# Helper functions for EDA
def suggest_encoding(series):
    unique_count = series.nunique()
    ordered_categories = ['low', 'medium', 'high']  # Customize as needed
    is_ordinal = any(str(cat).lower() in ordered_categories for cat in series.unique())
    
    if is_ordinal:
        return "Label Encoding (Ordinal Encoding)"
    elif unique_count <= 5:
        return "OneHot Encoding"
    else:
        return "Label/Ordinal Encoding (consider domain knowledge)"

def suggest_treatment(df):
    suggestions = []
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            dtype = df[col].dtype
            if dtype in ['int64', 'float64']:
                suggestions.append(f"{col}: Numerical - Consider mean/median imputation")
            else:
                suggestions.append(f"{col}: Categorical - Consider mode imputation or 'Missing' category")
    return "\n".join(suggestions) if suggestions else "No missing values found"

def check_scaling_type(df):
    scaling_recommendations = {}
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        if df[col].max() > 100 or df[col].min() < -100:
            scaling_recommendations[col] = "Standard Scaling (mean=0, std=1)"
        elif (df[col].max() > 10 and df[col].min() >= 0) or (df[col].max() > 1 and df[col].min() >= 0):
            scaling_recommendations[col] = "MinMax Scaling (0-1 range)"
        else:
            scaling_recommendations[col] = "No scaling needed"
    return scaling_recommendations


# Home Page
if st.session_state["page"] == "Home":
    st.markdown(
        """
        <style>
            .unique-title {
                text-align: center;
                font-size: 50px;
                font-weight: bold;
                text-transform: uppercase;
                color: #0047AB;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                background: linear-gradient(to right, #0047AB, #007BFF);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                letter-spacing: 2px;
                position: relative;
                top: -20px;
            }
        </style>
        <div class='unique-title'>AI-POWERED INTELLIGENCE SUITE</div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3,= st.columns(3)
    with col2:
        st.image(
            r"https://cdn-icons-png.flaticon.com/512/1118/1118881.png",
            use_container_width=True,
        )
        st.markdown('<div class="custom-button">', unsafe_allow_html=True)
        if st.button("Model Selection"):
            navigate_to("Model Selection")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="custom-button">', unsafe_allow_html=True)
        st.image(
            r"https://icon-library.com/images/ratings-icon/ratings-icon-5.jpg",
            use_container_width=True,
        )

        if st.button("Performance Rating"):
            navigate_to("Performance Rating")
        st.markdown('</div>', unsafe_allow_html=True)
    with col1:

        st.markdown('<div class="custom-button">', unsafe_allow_html=True)
        st.image(
                r"https://cdn-icons-png.flaticon.com/512/2998/2998250.png",
                use_container_width=True,
            )
        
        if st.button("EDA"):
            navigate_to("EDA")
        st.markdown('</div>', unsafe_allow_html=True)

# EDA Page
elif st.session_state["page"] == "EDA":
    st.markdown(
        """
        <style>
            .eda-title {
                text-align: center;
                font-size: 36px;
                font-weight: bold;
                text-transform: uppercase;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                position: relative;
                top: -10px;
                letter-spacing: 1px;
                background: linear-gradient(to right, #0047AB, #007BFF);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
        </style>
        <div class='eda-title'>Exploratory Data Analysis</div>
        """,
        unsafe_allow_html=True
    )

    st.write("Upload your dataset for comprehensive exploratory analysis")
    
    col3, col4 = st.columns(2)
    with col3:
        if st.button("🔙 Back"):
            navigate_to("Home")
    with col4:
        if st.button("🏠 Home"):
            navigate_to("Home")
    
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        with st.spinner("Loading dataset..."):
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.df = df
                st.success("Dataset loaded successfully!")
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        col1, col2 = st.columns(2)
        
        # Sidebar options
        with st.sidebar:
            st.header("EDA Options")
            
            dataprev = st.checkbox("Data Preview")
            Info = st.checkbox("Dataset Information")
            describe = st.checkbox("Describe")
            number_columns = st.checkbox("Show numerical & categorical Columns")
            duplicates = st.checkbox("Check for duplicates")
            Nulls = st.checkbox("Show null Values")
            outliers = st.checkbox("Show Outliers Columns")
            Encoding = st.checkbox("Encode")
            redundant_features = st.checkbox("Check Redundant Features")
            univariate = st.checkbox("Univariate Analysis")
            bivariate = st.checkbox("Bivariate Analysis")
            multivariate = st.checkbox("Multivariate Analysis")
            feature_selection = st.checkbox("Selecting Target & Features")
            feature_scaling = st.checkbox("Scaling Recommendation")
            Full_eda = st.checkbox("Automated EDA")
        
        # Main content based on selections
        if dataprev:
            st.header("DATA PREVIEW")
            st.dataframe(df)

        if Info:
            buffer = io.StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()
            st.text("Dataset Information:")
            st.text(info_str)

        if describe:
            st.header("DESCRIBE")
            desc = df.describe()
            st.dataframe(desc)
            st.write("Shape of Dataset")
            st.write(f"**Number of Columns:** {df.shape[1]}")
            st.write(f"**Number of Rows:** {df.shape[0]}")

        if number_columns:
            st.header("TYPE OF COLUMNS")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            datetime_cols = df.select_dtypes(include=[np.datetime64]).columns.tolist()
            
            if num_cols:
                st.write(f"**Numerical Columns:** {num_cols}")
            if cat_cols:
                st.write(f"**Categorical Columns:** {cat_cols}")
            if datetime_cols:
                st.write(f"**DateTime Columns:** {datetime_cols}")

        if duplicates:
            try:
                st.write("📊 Original DataFrame:")
                st.dataframe(df)

                duplicate_rows = df[df.duplicated()]
                duplicate_count = duplicate_rows.shape[0]

                if duplicate_count > 0:
                    st.warning(f"Found {duplicate_count} duplicate rows in the dataset!")
                    st.write("🔍 Duplicate Rows:")
                    st.dataframe(duplicate_rows)

                    if st.button("Remove Duplicates"):
                        df_cleaned = df.drop_duplicates().reset_index(drop=True)
                        st.session_state.df = df_cleaned
                        st.success("Duplicates removed successfully!")
                        st.write("🧹 Cleaned DataFrame (duplicates removed):")
                        st.dataframe(df_cleaned)
                else:
                    st.success("No duplicates found!")
            except Exception as e:
                st.error(f"Error checking duplicates: {str(e)}")

        if Nulls:
            null_counts = df.isnull().sum()
            total_values = df.shape[0]
            null_percentage = (null_counts / total_values) * 100

            st.write("### Null Percentage")
            null_summary = pd.DataFrame({
                "Missing Values": null_counts,
                "Percentage": null_percentage,
                "Data Type": df.dtypes,
            })
            st.dataframe(null_summary)

            st.write("### Suggested Treatments")
            st.write(suggest_treatment(df))

        if outliers:
            tab1, tab2 = st.tabs(["Multi", "Individual"])
            with tab1:
                num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
                if num_cols:
                    st.write("### 📊 Boxplots of Numerical Columns")
                    fig, axes = plt.subplots(nrows=(len(num_cols) // 3) + 1, ncols=3, figsize=(15, 5 * (len(num_cols) // 3 + 1)))
                    axes = axes.flatten()
                    for i, col in enumerate(num_cols):
                        sns.boxplot(x=df[col], ax=axes[i])
                        axes[i].set_title(col, fontsize=12)
                    for j in range(i + 1, len(axes)):
                        fig.delaxes(axes[j])
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning("No numerical columns found!")
            
            with tab2:
                num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
                if num_cols:
                    selected_col = st.selectbox("Select a numerical column:", num_cols)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.boxplot(x=df[selected_col], ax=ax)
                    ax.set_title(f"Boxplot of {selected_col}")
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning("No numerical columns found!")

        if Encoding:
            num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
            cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

            st.subheader("Column Types")
            st.write("Numerical Columns:", num_cols)
            st.write("Categorical Columns:", cat_cols)

            st.subheader("Encoding Recommendations")
            encoding_choices = {}
            for col in cat_cols:
                suggested_encoding = suggest_encoding(df[col])
                encoding_info = ""
                if suggested_encoding == "Label Encoding (Ordinal Encoding)":
                    encoding_info = "\n📊 **The Feature Has a Natural Order (Ordinal Data)**"
                elif suggested_encoding == "OneHot Encoding":
                    encoding_info = "\n📊 **No Natural Order (Nominal Data)**"
                
                encoding_choices[col] = st.selectbox(
                    f"Encoding method for '{col}' (suggested: {suggested_encoding}){encoding_info}",
                    ["None", "OneHot Encoding", "Label Encoding (Ordinal Encoding)", "Label/Ordinal Encoding"],
                    index=["None", "OneHot Encoding", "Label Encoding (Ordinal Encoding)", "Label/Ordinal Encoding"].index(suggested_encoding),
                )

        if redundant_features:
            tab1, tab2 = st.tabs(["Feature Redundancy Detection", "Correlation"])
            with tab1:
                st.subheader("Redundant Feature Detection")
                df_encoded = df.copy()
                categorical_cols = df_encoded.select_dtypes(exclude=[np.number]).columns.tolist()
                for col in categorical_cols:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                
                corr_matrix = df_encoded.corr().abs()
                upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.85)]
                
                if high_corr_features:
                    st.write("Highly correlated features (correlation > 0.85):", high_corr_features)
                else:
                    st.success("No highly correlated features found.")
            
            with tab2:
                num_df = df.select_dtypes(include=["int64", "float64"])
                if not num_df.empty:
                    corr_matrix = num_df.corr()
                    fig, ax = plt.subplots(figsize=(12, 8))
                    sns.heatmap(data=corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning("No numerical columns for correlation analysis!")

        if univariate:
            selected_col = st.selectbox("Select a column for Univariate Analysis", df.columns)
            if df[selected_col].dtype in ["int64", "float64"]:
                fig, ax = plt.subplots()
                sns.histplot(df[selected_col], bins=20, kde=True, ax=ax)
                st.pyplot(fig)
                plt.close(fig)
            else:
                fig, ax = plt.subplots()
                sns.countplot(x=df[selected_col], ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)
                plt.close(fig)

        if bivariate:
            st.header("Bivariate Analysis Explorer")
            target_column = st.selectbox("Select Target Column", df.columns)
            feature_column = st.selectbox("Select Feature Column", df.columns)
            
            if target_column and feature_column:
                target_type = "categorical" if df[target_column].dtype == "object" else "numerical"
                feature_type = "categorical" if df[feature_column].dtype == "object" else "numerical"
                
                if target_type == "numerical" and feature_type == "numerical":
                    chart_type = "Scatter Plot"
                elif target_type == "numerical" and feature_type == "categorical":
                    chart_type = "Box Plot"
                elif target_type == "categorical" and feature_type == "numerical":
                    chart_type = "Distribution Plot"
                else:
                    chart_type = "Count Plot"
                
                st.write(f"🔍 Recommended Chart: **{chart_type}**")
                
                fig, ax = plt.subplots(figsize=(8, 5))
                if chart_type == "Scatter Plot":
                    sns.scatterplot(x=df[feature_column], y=df[target_column], ax=ax)
                elif chart_type == "Box Plot":
                    sns.boxplot(x=df[feature_column], y=df[target_column], ax=ax)
                elif chart_type == "Distribution Plot":
                    sns.histplot(data=df, x=feature_column, hue=target_column, kde=True, bins=30, ax=ax)
                elif chart_type == "Count Plot":
                    sns.countplot(x=df[feature_column], hue=df[target_column], ax=ax)
                
                st.pyplot(fig)
                plt.close(fig)

        if multivariate:
            st.header("Multivariate Analysis")
            selected_cols = st.multiselect("Select columns for analysis", df.columns)
            if len(selected_cols) > 1:
                with st.spinner("Generating pairplot..."):
                    fig = sns.pairplot(df[selected_cols])
                    st.pyplot(fig)
                    plt.close()

        if feature_selection:
            st.header("Feature Selection")
            target_column = st.selectbox("Select Target Column", df.columns)
            if target_column:
                if df[target_column].dtype in ["int64", "float64"]:
                    st.subheader("Numerical Target Analysis")
                    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
                    if len(num_cols) > 1:
                        corr = df[num_cols].corr()[target_column].sort_values(ascending=False)
                        st.dataframe(corr)
                else:
                    st.subheader("Categorical Target Analysis")
                    st.write("Target distribution:")
                    fig, ax = plt.subplots()
                    sns.countplot(x=df[target_column], ax=ax)
                    st.pyplot(fig)
                    plt.close(fig)

        if feature_scaling:
            scaling_recommendations = check_scaling_type(df)
            st.header("Scaling Recommendations")
            for feature, scale_type in scaling_recommendations.items():
                st.write(f"{feature}: {scale_type}")

        if Full_eda:
            st.subheader("Pandas Profiling Report")
            with st.spinner("Generating profiling report..."):
                if df.shape[0] > 10000:
                    st.warning("Large dataset detected. Sampling 10,000 rows for faster processing.")
                    df_sample = df.sample(10000, random_state=42)
                else:
                    df_sample = df
                profile = ProfileReport(df_sample, title="Pandas Profiling Report", explorative=True)
                st_profile_report(profile)

# Model Selection Page
elif st.session_state["page"] == "Model Selection":
    st.markdown(
        """
        <style>
            .model-selection-title {
                text-align: center;
                font-size: 36px;
                font-weight: bold;
                text-transform: uppercase;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                position: relative;
                top: -10px;
                letter-spacing: 1px;
                background: linear-gradient(to right, #0047AB, #007BFF);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
        </style>
        <div class='model-selection-title'>Model Selection</div>
        """,
        unsafe_allow_html=True
    )

    st.write("Selecting algorithm for model training")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔙 Back"):
            navigate_to("Home")
    with col2:
        if st.button("🏠 Home"):
            navigate_to("Home")

    tab1, tab2, tab3 = st.tabs(["Model Training & Evaluation", "Cross Validation", "Tuning/Generalising"])
    
    with tab1:
        st.title("Automated Model Selection & Evaluation")
        
        # File uploaders
        features_file = st.file_uploader("Upload your CSV file for features", type=["csv"], key="features")
        target_file = st.file_uploader("Upload your CSV file for target", type=["csv"], key="target")

        if features_file and target_file:
            with st.spinner("Loading data..."):
                df_features = pd.read_csv(features_file)
                df_target = pd.read_csv(target_file)

            # Select target column
            target_column = st.selectbox("Select target column:", df_target.columns)

            if target_column:
                # Select feature columns
                selected_features = st.multiselect(
                    "Select feature columns (remove unwanted features):",
                    df_features.select_dtypes(include=[np.number]).columns,
                    default=[col for col in df_features.select_dtypes(include=[np.number]).columns if col != target_column],
                )

                X = df_features[selected_features]
                y = df_target[target_column]

                # Check if target needs encoding (categorical)
                if y.dtype == 'object' or y.nunique() <= 10:
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y)
                    st.info("Categorical target detected - Applied Label Encoding.")
                    y = pd.Series(y_encoded, name=target_column)
                    class_names = le.classes_
                else:
                    st.info("Numerical target detected - No encoding needed.")
                    class_names = None

                # Determine problem type
                unique_values = y.nunique()
                is_categorical = y.dtype == "object" or unique_values <= 10
                is_continuous = y.dtype in ["int64", "float64"] and unique_values > 10

                classification_checkbox = st.checkbox("Classification", value=is_categorical)
                regression_checkbox = st.checkbox("Regression", value=is_continuous)

                if classification_checkbox and not regression_checkbox:
                    target_type = "classification"
                elif regression_checkbox and not classification_checkbox:
                    target_type = "regression"
                else:
                    st.warning("Please select only one model type (Classification or Regression).")
                    st.stop()

                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                st.info(f"Automatically detected target type: {target_type.capitalize()}")
                st.session_state.target_type = target_type

                # Model selection and evaluation
                st.session_state.best_model = None
                st.session_state.best_score = -np.inf
                st.session_state.best_model_name = ""

                if target_type == "classification":
                    models = {
                        "Logistic Regression": LogisticRegression(max_iter=1000),
                        "Decision Tree Classifier": DecisionTreeClassifier(),
                        "Random Forest Classifier": RandomForestClassifier(),
                        "XGBoost Classifier": XGBClassifier(),
                        "Naive Bayes Classifier": GaussianNB(),
                        "SVM Classifier": SVC(probability=True),
                    }
                else:
                    models = {
                        "Linear Regression": LinearRegression(),
                        "Polynomial Regression": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
                        "Ridge Regression": Ridge(),
                        "Lasso Regression": Lasso(),
                        "ElasticNet Regression": ElasticNet(),
                        "SVM Regression": SVR(),
                        "Decision Tree Regressor": DecisionTreeRegressor(),
                        "Random Forest Regressor": RandomForestRegressor(),
                        "XGBoost Regressor": XGBRegressor(),
                    }

                for model_name, model in models.items():
                    try:
                        with st.expander(f"Model: {model_name}"):
                            with st.spinner(f"Training {model_name}..."):
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)

                            if target_type == "classification":
                                train_score = model.score(X_train, y_train)
                                test_score = model.score(X_test, y_test)
                                score_diff = abs(train_score - test_score)
                                fit_status = "Good Fit" if score_diff <= 0.05 else ("Overfit" if train_score > test_score else "Underfit")

                                # Calculate metrics
                                precision = precision_score(y_test, y_pred, average='weighted')
                                recall = recall_score(y_test, y_pred, average='weighted')
                                f1 = f1_score(y_test, y_pred, average='weighted')

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Train Score", f"{train_score * 100:.2f}%")
                                with col2:
                                    st.metric("Test Score", f"{test_score * 100:.2f}%")
                                with col3:
                                    st.metric("Fit Status", fit_status)

                                st.subheader("Classification Metrics")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Precision", f"{precision * 100:.2f}%")
                                with col2:
                                    st.metric("Recall", f"{recall * 100:.2f}%")
                                with col3:
                                    st.metric("F1 Score", f"{f1 * 100:.2f}%")

                                # Classification Report
                                st.subheader("Classification Report")
                                classification_rep = classification_report(y_test, y_pred, output_dict=True)
                                classification_df = pd.DataFrame(classification_rep).transpose()
                                st.table(classification_df)

                                # Confusion Matrix
                                st.subheader("Confusion Matrix")
                                conf_matrix = confusion_matrix(y_test, y_pred)
                                fig, ax = plt.subplots()
                                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                                ax.set_xlabel('Predicted')
                                ax.set_ylabel('Actual')
                                ax.set_title('Confusion Matrix')
                                st.pyplot(fig)
                                plt.close(fig)

                            else:  # Regression
                                train_score = model.score(X_train, y_train)
                                test_score = model.score(X_test, y_test)
                                score_diff = abs(train_score - test_score)
                                fit_status = "Good Fit" if score_diff <= 0.05 else ("Overfit" if train_score > test_score else "Underfit")

                                r2 = r2_score(y_test, y_pred)
                                mse = mean_squared_error(y_test, y_pred)
                                rmse = np.sqrt(mse)
                                mae = mean_absolute_error(y_test, y_pred)

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Train Score", f"{train_score * 100:.2f}%")
                                with col2:
                                    st.metric("Test Score", f"{test_score * 100:.2f}%")
                                with col3:
                                    st.metric("Fit Status", fit_status)

                                st.subheader("Regression Metrics")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("R² Score", f"{r2 * 100:.2f}%")
                                with col2:
                                    st.metric("RMSE", f"{rmse:.2f}")
                                with col3:
                                    st.metric("MAE", f"{mae:.2f}")

                                # Scatter plot for actual vs predicted values
                                st.subheader("Actual vs Predicted Values")
                                fig, ax = plt.subplots()
                                ax.scatter(y_test, y_pred, alpha=0.5)
                                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
                                ax.set_xlabel('Actual')
                                ax.set_ylabel('Predicted')
                                ax.set_title('Actual vs Predicted')
                                st.pyplot(fig)
                                plt.close(fig)

                            # Update best model
                            current_score = test_score if target_type == "classification" else r2
                            if current_score > st.session_state.best_score:
                                st.session_state.best_score = current_score
                                st.session_state.best_model = model
                                st.session_state.best_model_name = model_name

                    except Exception as e:
                        st.error(f"Failed to train {model_name}: {str(e)}")
                        continue

                # Save the best model
                if st.session_state.best_model:
                    st.success(f"Best Model: {st.session_state.best_model_name} with score: {st.session_state.best_score * 100:.2f}%")
                    if st.button("Save Best Model"):
                        joblib.dump(st.session_state.best_model, 'best_model.pkl')
                        st.success("Best model saved as 'best_model.pkl'")

    with tab2:
        st.header("Cross-Validation")
        if st.session_state.best_model is not None:
            cv_method = st.selectbox("Choose Cross-Validation Method:", ["K-Fold", "Stratified K-Fold"])
            cv_folds = st.slider("Select number of folds:", 2, 10, 5)
            scoring_metric = st.selectbox(
                "Select scoring metric:",
                ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
                if st.session_state.target_type == "classification" else
                ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]
            )
            
            if st.button("Run Cross-Validation"):
                with st.spinner("Running cross-validation..."):
                    try:
                        cv = KFold(n_splits=cv_folds) if cv_method == "K-Fold" else StratifiedKFold(n_splits=cv_folds)
                        scores = cross_val_score(st.session_state.best_model, X_scaled, y, cv=cv, scoring=scoring_metric)
                        score_std = np.std(scores)

                        st.info(f"Cross-Validation Scores: {scores}")
                        st.success(f"Mean Score: {np.mean(scores) * 100:.2f}%")
                        st.info(f"Standard Deviation: {score_std * 100:.2f}%")

                        if score_std * 100 < 2:
                            st.success("Low variance — Model is stable and generalizing well!")
                        elif score_std * 100 <= 5:
                            st.warning("Moderate variance — Model might be slightly sensitive to data splits.")
                        else:
                            st.error("High variance — Model might be unstable. Consider tuning or regularization.")
                    except Exception as e:
                        st.error(f"Error during cross-validation: {str(e)}")
        else:
            st.warning("Please train models first in the 'Model Training & Evaluation' tab")

    with tab3:
        st.header("Hyperparameter Tuning")
        if st.session_state.best_model is not None:
            model_name = st.session_state.best_model_name
            st.write(f"Tuning {model_name}")
            
            if "Random Forest" in model_name:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                }
                grid_search = GridSearchCV(st.session_state.best_model, param_grid, cv=3, n_jobs=-1)
                if st.button("Run Grid Search"):
                    with st.spinner("Running grid search..."):
                        grid_search.fit(X_scaled, y)
                        st.success(f"Best parameters: {grid_search.best_params_}")
                        st.session_state.best_model = grid_search.best_estimator_
                        st.session_state.best_score = grid_search.best_score_
                        st.write(f"Updated best score: {st.session_state.best_score * 100:.2f}%")
            else:
                st.info("Tuning not implemented for this model yet.")
        else:
            st.warning("Please train models first in the 'Model Training & Evaluation' tab")

# Performance Rating Page
elif st.session_state["page"] == "Performance Rating":
    st.markdown(
        """
        <style>
            .eda-title {
                text-align: center;
                font-size: 36px;
                font-weight: bold;
                text-transform: uppercase;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                position: relative;
                top: -10px;
                letter-spacing: 1px;
                background: linear-gradient(to right, #0047AB, #007BFF);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
        </style>
        <div class='eda-title'>Performance Rating</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h4 style='text-align: center;'>Predict employee performance rating</h4>", unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    with col3:
        if st.button("🔙 Back"):
            navigate_to("Home")
    with col4:
        if st.button("🏠 Home"):
            navigate_to("Home")
    
    try:
        
        ohe_for_job_sat = joblib.load('onehotforjobsat.pkl')
        label_for_job_sat = joblib.load('labelforjobsat.pkl')
        logistic_model = joblib.load('logistic_for_performance_rating.pkl')
        
        if None in [ohe_for_job_sat, label_for_job_sat, logistic_model]:
            raise FileNotFoundError("One or more model files could not be loaded.")

        # Input Form
        with st.form("perf_form"):
            col1, col2 = st.columns(2)

            with col1:
                Age = st.number_input("Age", min_value=18, max_value=60, step=1)
                DistanceFromHome = st.number_input("Distance from Home", min_value=0, step=1)
                Education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
                EnvironmentSatisfaction = st.slider("Environment Satisfaction", 1, 4)
                JobInvolvement = st.slider("Job Involvement", 1, 4)
                MonthlyIncome = st.number_input("Monthly Income", min_value=0, step=100)
                MonthlyRate = st.number_input("Monthly Rate", min_value=0, step=100)

            with col2:
                PercentSalaryHike = st.slider("Percent Salary Hike", 0, 100)
                RelationshipSatisfaction = st.slider("Relationship Satisfaction", 1, 4)
                WorkLifeBalance = st.slider("Work-Life Balance", 1, 4)
                YearsInCurrentRole = st.number_input("Years in Current Role", min_value=0, step=1)
                OverTime = st.selectbox("OverTime", ["Yes", "No"])
                
                # Categorical inputs
                BusinessTravel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Frequently", "Travel_Rarely"])
                Department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
                EducationField = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
                Gender = st.selectbox("Gender", ["Male", "Female"])
                JobRole = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", 
                                            "Healthcare Representative", "Manager", "Sales Representative", 
                                            "Research Director", "Human Resources"])
                MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

            submitted = st.form_submit_button("Predict")

        if submitted:
            # Input validation
            if any([
                Age < 18 or Age > 60,
                DistanceFromHome < 0,
                MonthlyIncome < 0,
                MonthlyRate < 0,
                YearsInCurrentRole < 0,
            ]):
                st.error("Invalid input values. Please check your inputs (e.g., no negative values, age between 18-60).")
            else:
                with st.spinner("Making prediction..."):
                    # Numerical inputs
                    numerical_inputs = [
                        Age,
                        DistanceFromHome,
                        Education,
                        EnvironmentSatisfaction,
                        JobInvolvement,
                        MonthlyIncome,
                        MonthlyRate,
                        PercentSalaryHike,
                        RelationshipSatisfaction,
                        WorkLifeBalance,
                        YearsInCurrentRole,
                        label_for_job_sat.transform([[OverTime]])[0],
                    ]

                    # Categorical inputs
                    cat_input = pd.DataFrame([{
                        "BusinessTravel": BusinessTravel,
                        "Department": Department,
                        "EducationField": EducationField,
                        "Gender": Gender,
                        "JobRole": JobRole,
                        "MaritalStatus": MaritalStatus
                    }])

                    cat_encoded = ohe_for_job_sat.transform(cat_input)

                    # Final input
                    final_input = np.concatenate([np.array(numerical_inputs).reshape(1, -1), cat_encoded], axis=1)

                    prediction = logistic_model.predict(final_input)[0]

                    # Map prediction to label
                    rating_label = {
                        3: "Excellent (Exceeds Expectations)",
                        4: "Outstanding (Top Performer)"
                    }.get(prediction, "Unknown")

                    # Set background gradient by rating
                    bg_gradient = "linear-gradient(to right, #3A7BD5, #00d2ff);" if prediction == 3 else "linear-gradient(to right, #8E2DE2, #4A00E0);"

                    # Styled block with CSS
                    st.markdown(f"""
                        <div style="
                            padding: 1.5rem;
                            border-radius: 15px;
                            background: {bg_gradient};
                            color: white;
                            text-align: center;
                            font-size: 22px;
                            font-weight: bold;
                            margin-top: 20px;
                            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.15);
                        ">
                            Predicted Performance Rating: {prediction} - {rating_label}
                        </div>
                    """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error in performance rating prediction: {str(e)}")

# Placeholder for Employee Attrition Page
elif st.session_state["page"] == "Predicting Employee Attrition":
    st.title("Predicting Employee Attrition")
    st.write("This feature is not implemented yet.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔙 Back"):
            navigate_to("Home")
    with col2:
        if st.button("🏠 Home"):
            navigate_to("Home")