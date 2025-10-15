import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap
from lime.lime_tabular import LimeTabularExplainer
import warnings

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="NVIDIA Competitor Analysis",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #76B900;
        font-weight: bold;
    }
    h2, h3 {
        color: #333;
    }
    .highlight-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #76B900;
        border: 1px solid #e0e0e0;
        margin: 20px 0;
        color: #333333 !important;
    }
    .highlight-box h2 {
        color: #76B900 !important;
        margin-top: 0;
        margin-bottom: 15px;
    }
    .highlight-box p {
        color: #333333 !important;
        line-height: 1.6;
        font-size: 16px;
    }
    .highlight-box li {
        color: #333333 !important;
        line-height: 1.6;
        margin-bottom: 8px;
    }
    .highlight-box ul {
        padding-left: 20px;
        margin-bottom: 0;
    }
    /* Reduce sidebar width */
    section[data-testid="stSidebar"] {
        width: 250px !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load data with caching
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset.csv")
    return df


# Prepare data for modeling
@st.cache_data
def prepare_data(df, target_column):
    df_model = df.copy()

    # Encode categorical variables
    le_dict = {}
    for col in df_model.select_dtypes(include=["object"]).columns:
        if col != target_column:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            le_dict[col] = le

    # Separate features and target
    if target_column in df_model.columns:
        X = df_model.drop(columns=[target_column])
        y = df_model[target_column]
    else:
        X = df_model
        y = None

    return X, y, le_dict


# Train models
@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=42
        ),
    }

    trained_models = {}
    metrics = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        trained_models[name] = model
        metrics[name] = {
            "R² Score": r2_score(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAE": mean_absolute_error(y_test, y_pred),
            "y_test": y_test,
            "y_pred": y_pred,
        }

    return trained_models, metrics, scaler, X_train, X_test, y_train, y_test


# Main app
def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Overview",
            "Regression Prediction",
            "Data Insights",
            "Model Interpretability",
        ],
    )

    # Load data
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading dataset.csv: {e}")
        st.info("Please ensure 'dataset.csv' is in the same directory as this script.")
        return

    # Page routing
    if page == "Overview":
        show_overview(df)
    elif page == "Regression Prediction":
        show_prediction(df)
    elif page == "Data Insights":
        show_insights(df)
    elif page == "Model Interpretability":
        show_interpretability(df)


def show_overview(df):
    st.title("NVIDIA Competitor Analysis Dashboard")

    st.markdown("---")

    # Problem Statement
    st.markdown(
        """
    <div class="highlight-box">
    <h2>The Challenge</h2>
    <p>The semiconductor industry is highly competitive, with companies like NVIDIA, AMD, Intel, Apple, and Broadcom
    vying for market dominance across gaming GPUs, data centers, and AI compute segments. Understanding the competitive
    landscape requires analyzing multiple factors including market share, revenue, product performance, power efficiency,
    and innovation focus.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("##")

    # Solution
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(
            """
        <div class="highlight-box">
        <h2>Our Solution</h2>
        <p>This interactive dashboard provides comprehensive analysis through:</p>
        <ul>
            <li><strong>Predictive Analytics:</strong> Machine learning models to predict key performance metrics</li>
            <li><strong>Data Visualization:</strong> Interactive charts and graphs revealing market trends</li>
            <li><strong>Model Transparency:</strong> Explainable AI techniques to understand prediction drivers</li>
            <li><strong>Comparative Analysis:</strong> Side-by-side comparison of competitor strengths and weaknesses</li>
        </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="highlight-box">
        <h2>Key Features</h2>
        <ul>
            <li><strong>Regression Prediction:</strong> Adjust parameters to forecast outcomes</li>
            <li><strong>Data Insights:</strong> Explore data distributions and relationships</li>
            <li><strong>Model Interpretability:</strong> SHAP and LIME analysis for feature importance</li>
            <li><strong>Performance Metrics:</strong> Compare accuracy across multiple models</li>
        </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("##")

    # Dataset Overview
    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; padding: 5px; border-radius: 10px; text-align: center;">
                <h3 style="color: #76B900; margin: 0; font-size: 18px;">Total Companies</h3>
                <p style="color: #333; font-size: 32px; font-weight: bold; margin: 5px 0 0 0;">{len(df)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; padding: 5px; border-radius: 10px; text-align: center;">
                <h3 style="color: #76B900; margin: 0; font-size: 18px;">Features</h3>
                <p style="color: #333; font-size: 32px; font-weight: bold; margin: 5px 0 0 0;">{len(df.columns)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; padding: 5px; border-radius: 10px; text-align: center;">
                <h3 style="color: #76B900; margin: 0; font-size: 18px;">Numerical Columns</h3>
                <p style="color: #333; font-size: 32px; font-weight: bold; margin: 5px 0 0 0;">{len(df.select_dtypes(include=[np.number]).columns)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; padding: 5px; border-radius: 10px; text-align: center;">
                <h3 style="color: #76B900; margin: 0; font-size: 18px;">Categorical Columns</h3>
                <p style="color: #333; font-size: 32px; font-weight: bold; margin: 5px 0 0 0;">{len(df.select_dtypes(include=["object"]).columns)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("##")

    # Data Preview
    st.subheader("Data Preview")
    st.dataframe(df, use_container_width=True, height=300)

    st.markdown("##")

    # Quick Statistics
    st.subheader("Quick Statistics")
    st.dataframe(df.describe(), use_container_width=True)


def show_prediction(df):
    st.title("Regression Prediction")

    st.markdown(
        """
    Adjust the input parameters below to see how they affect the predicted output.
    The model uses historical data to make predictions based on your selections.
    """
    )

    st.markdown("---")

    # Select target variable
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    col1, col2 = st.columns([1, 3])

    with col1:
        target_col = st.selectbox("Select Target Variable to Predict", numerical_cols)

    with col2:
        model_choice = st.selectbox(
            "Select Prediction Model",
            ["Linear Regression", "Random Forest", "Gradient Boosting"],
        )

    st.markdown("##")

    # Prepare data
    X, y, le_dict = prepare_data(df, target_col)

    if y is not None:
        # Train models
        trained_models, metrics, scaler, X_train, X_test, y_train, y_test = (
            train_models(X, y)
        )

        # Display model performance
        st.subheader(f"Model Performance: {model_choice}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #76B900; margin: 0; font-size: 18px;">R² Score</h3>
                    <p style="color: #333; font-size: 32px; font-weight: bold; margin: 10px 0 0 0;">{metrics[model_choice]['R² Score']:.4f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #76B900; margin: 0; font-size: 18px;">RMSE</h3>
                    <p style="color: #333; font-size: 32px; font-weight: bold; margin: 10px 0 0 0;">{metrics[model_choice]['RMSE']:.4f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #76B900; margin: 0; font-size: 18px;">MAE</h3>
                    <p style="color: #333; font-size: 32px; font-weight: bold; margin: 10px 0 0 0;">{metrics[model_choice]['MAE']:.4f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Input section
        st.subheader("Adjust Input Parameters")

        input_data = {}

        # Create columns for input fields
        feature_cols = st.columns(3)

        for idx, col in enumerate(X.columns):
            with feature_cols[idx % 3]:
                if col in df.select_dtypes(include=[np.number]).columns:
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    mean_val = float(df[col].mean())
                    input_data[col] = st.slider(
                        col,
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        step=(max_val - min_val) / 100,
                    )
                else:
                    unique_vals = df[col].unique()
                    selected = st.selectbox(col, unique_vals)
                    if col in le_dict:
                        input_data[col] = le_dict[col].transform([selected])[0]
                    else:
                        input_data[col] = selected

        st.markdown("##")

        # Make prediction
        if st.button("Generate Prediction", type="primary"):
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)

            prediction = trained_models[model_choice].predict(input_scaled)[0]

            st.markdown("##")
            st.success("Prediction Complete!")

            # Display prediction
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                st.markdown(
                    f"""
                <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 15px; color: white;">
                    <h2>Predicted {target_col}</h2>
                    <h1 style="font-size: 48px; margin: 20px 0;">{prediction:.2f}</h1>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            st.markdown("##")

            # Show comparison with actual values
            st.subheader("Prediction Context")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(
                    f"""
                    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
                        <h3 style="color: #76B900; margin: 0; font-size: 18px;">Minimum in Dataset</h3>
                        <p style="color: #333; font-size: 32px; font-weight: bold; margin: 10px 0 0 0;">{df[target_col].min():.2f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f"""
                    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
                        <h3 style="color: #76B900; margin: 0; font-size: 18px;">Average in Dataset</h3>
                        <p style="color: #333; font-size: 32px; font-weight: bold; margin: 10px 0 0 0;">{df[target_col].mean():.2f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col3:
                st.markdown(
                    f"""
                    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
                        <h3 style="color: #76B900; margin: 0; font-size: 18px;">Maximum in Dataset</h3>
                        <p style="color: #333; font-size: 32px; font-weight: bold; margin: 10px 0 0 0;">{df[target_col].max():.2f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def show_insights(df):
    st.title("Data Insights")

    st.markdown(
        "Explore comprehensive visualizations of the competitor analysis dataset."
    )

    st.markdown("---")

    # Data Distribution
    st.header("Data Distributions")

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    selected_col = st.selectbox(
        "Select Column for Distribution Analysis", numerical_cols
    )

    col1, col2 = st.columns(2)

    with col1:
        # Histogram
        fig = px.histogram(
            df,
            x=selected_col,
            title=f"Distribution of {selected_col}",
            color_discrete_sequence=["#76B900"],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Box plot
        fig = px.box(
            df,
            y=selected_col,
            title=f"Box Plot of {selected_col}",
            color_discrete_sequence=["#76B900"],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Market Share Analysis
    st.header("Market Share Analysis")

    col1, col2 = st.columns(2)

    with col1:
        if "Company" in df.columns and "Market Share Gaming GPU (%)" in df.columns:
            fig = px.pie(
                df,
                values="Market Share Gaming GPU (%)",
                names="Company",
                title="Gaming GPU Market Share Distribution",
                color_discrete_sequence=px.colors.sequential.Greens,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "Company" in df.columns and "Market Share Data Center GPU (%)" in df.columns:
            fig = px.pie(
                df,
                values="Market Share Data Center GPU (%)",
                names="Company",
                title="Data Center GPU Market Share Distribution",
                color_discrete_sequence=px.colors.sequential.Blues,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Performance Comparison
    st.header("Performance Comparison")

    if "Company" in df.columns:
        performance_cols = [
            col for col in numerical_cols if "Performance" in col or "Power" in col
        ]

        if performance_cols:
            selected_metrics = st.multiselect(
                "Select Metrics to Compare",
                performance_cols,
                default=(
                    performance_cols[:2]
                    if len(performance_cols) >= 2
                    else performance_cols
                ),
            )

            if selected_metrics:
                fig = go.Figure()

                for metric in selected_metrics:
                    fig.add_trace(go.Bar(name=metric, x=df["Company"], y=df[metric]))

                fig.update_layout(
                    title="Company Performance Metrics Comparison",
                    barmode="group",
                    xaxis_title="Company",
                    yaxis_title="Score",
                    height=500,
                )

                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Correlation Heatmap
    st.header("Feature Correlation Heatmap")

    corr_matrix = df[numerical_cols].corr()

    fig = px.imshow(
        corr_matrix,
        title="Correlation Matrix of Numerical Features",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        labels=dict(color="Correlation"),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Revenue Analysis
    st.header("Revenue Analysis")

    if "Company" in df.columns and "Annual Revenue (Billion USD)" in df.columns:
        fig = px.bar(
            df.sort_values("Annual Revenue (Billion USD)", ascending=True),
            y="Company",
            x="Annual Revenue (Billion USD)",
            title="Annual Revenue by Company",
            orientation="h",
            color="Annual Revenue (Billion USD)",
            color_continuous_scale="Viridis",
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Scatter Plot Analysis
    st.header("Relationship Analysis")

    col1, col2 = st.columns(2)

    with col1:
        x_axis = st.selectbox("Select X-axis", numerical_cols, key="scatter_x")
    with col2:
        y_axis = st.selectbox(
            "Select Y-axis",
            numerical_cols,
            key="scatter_y",
            index=1 if len(numerical_cols) > 1 else 0,
        )

    if "Company" in df.columns:
        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            color="Company",
            size=df[numerical_cols[0]],
            title=f"{y_axis} vs {x_axis}",
            hover_data=df.columns,
        )

        st.plotly_chart(fig, use_container_width=True)


def show_interpretability(df):
    st.title("Model Interpretability")

    st.markdown(
        "Understand how models make predictions through feature importance analysis and performance comparison."
    )

    st.markdown("---")

    # Select target variable
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    target_col = st.selectbox("Select Target Variable for Analysis", numerical_cols)

    # Prepare data
    X, y, le_dict = prepare_data(df, target_col)

    if y is not None:
        # Train models
        trained_models, metrics, scaler, X_train, X_test, y_train, y_test = (
            train_models(X, y)
        )

        st.markdown("##")

        # Model Performance Comparison
        st.header("Model Performance Comparison")

        metrics_df = pd.DataFrame(
            {
                "Model": list(metrics.keys()),
                "R² Score": [metrics[m]["R² Score"] for m in metrics.keys()],
                "RMSE": [metrics[m]["RMSE"] for m in metrics.keys()],
                "MAE": [metrics[m]["MAE"] for m in metrics.keys()],
            }
        )

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                metrics_df,
                x="Model",
                y="R² Score",
                title="R² Score Comparison (Higher is Better)",
                color="R² Score",
                color_continuous_scale="Greens",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                metrics_df,
                x="Model",
                y="RMSE",
                title="RMSE Comparison (Lower is Better)",
                color="RMSE",
                color_continuous_scale="Reds_r",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Metrics table
        st.subheader("Detailed Metrics")
        st.dataframe(
            metrics_df.style.highlight_max(
                subset=["R² Score"], color="lightgreen"
            ).highlight_min(subset=["RMSE", "MAE"], color="lightgreen"),
            use_container_width=True,
        )

        st.markdown("---")

        # Actual vs Predicted
        st.header("Prediction Accuracy Visualization")

        model_for_viz = st.selectbox(
            "Select Model for Visualization", list(trained_models.keys())
        )

        y_test_viz = metrics[model_for_viz]["y_test"]
        y_pred_viz = metrics[model_for_viz]["y_pred"]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=y_test_viz,
                y=y_pred_viz,
                mode="markers",
                name="Predictions",
                marker=dict(size=10, color="#76B900", opacity=0.6),
            )
        )

        # Perfect prediction line
        min_val = min(y_test_viz.min(), y_pred_viz.min())
        max_val = max(y_test_viz.max(), y_pred_viz.max())

        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Perfect Prediction",
                line=dict(color="red", dash="dash"),
            )
        )

        fig.update_layout(
            title=f"Actual vs Predicted Values - {model_for_viz}",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # SHAP Analysis
        st.header("SHAP Feature Importance Analysis")

        st.markdown(
            """
        SHAP (SHapley Additive exPlanations) values show how much each feature contributes to the model's predictions.
        Positive values push predictions higher, negative values push them lower.
        """
        )

        model_for_shap = st.selectbox(
            "Select Model for SHAP Analysis",
            list(trained_models.keys()),
            key="shap_model",
        )

        with st.spinner("Generating SHAP analysis..."):
            try:
                # Use a sample of data for SHAP to speed up computation
                X_train_sample = X_train.iloc[:100] if len(X_train) > 100 else X_train
                X_train_scaled_sample = scaler.transform(X_train_sample)

                if model_for_shap == "Linear Regression":
                    explainer = shap.LinearExplainer(
                        trained_models[model_for_shap], X_train_scaled_sample
                    )
                    shap_values = explainer.shap_values(X_train_scaled_sample)
                else:
                    explainer = shap.Explainer(
                        trained_models[model_for_shap], X_train_scaled_sample
                    )
                    shap_values = explainer(X_train_scaled_sample)
                    if hasattr(shap_values, "values"):
                        shap_values = shap_values.values

                # Feature importance bar plot
                shap_importance = np.abs(shap_values).mean(axis=0)
                feature_importance_df = pd.DataFrame(
                    {"Feature": X.columns, "Importance": shap_importance}
                ).sort_values("Importance", ascending=True)

                fig = px.bar(
                    feature_importance_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="SHAP Feature Importance",
                    color="Importance",
                    color_continuous_scale="Viridis",
                )

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(f"Could not generate SHAP analysis: {e}")

        st.markdown("---")

        # LIME Analysis
        st.header("LIME Feature Importance Analysis")

        st.markdown(
            """
        LIME (Local Interpretable Model-agnostic Explanations) explains individual predictions
        by approximating the model locally with an interpretable model.
        """
        )

        model_for_lime = st.selectbox(
            "Select Model for LIME Analysis",
            list(trained_models.keys()),
            key="lime_model",
        )

        instance_idx = st.slider("Select Instance to Explain", 0, len(X_test) - 1, 0)

        with st.spinner("Generating LIME explanation..."):
            try:
                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                explainer = LimeTabularExplainer(
                    X_train_scaled,
                    feature_names=X.columns.tolist(),
                    mode="regression",
                    random_state=42,
                )

                exp = explainer.explain_instance(
                    X_test_scaled[instance_idx],
                    trained_models[model_for_lime].predict,
                    num_features=len(X.columns),
                )

                # Extract LIME values
                lime_values = exp.as_list()
                lime_df = pd.DataFrame(lime_values, columns=["Feature", "Importance"])
                lime_df = lime_df.sort_values("Importance", ascending=True)

                fig = px.bar(
                    lime_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title=f"LIME Feature Importance for Instance {instance_idx}",
                    color="Importance",
                    color_continuous_scale="RdYlGn",
                )

                st.plotly_chart(fig, use_container_width=True)

                # Show actual prediction
                actual_value = y_test.iloc[instance_idx]
                predicted_value = trained_models[model_for_lime].predict(
                    X_test_scaled[instance_idx].reshape(1, -1)
                )[0]

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"""
                        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
                            <h3 style="color: #76B900; margin: 0; font-size: 18px;">Actual Value</h3>
                            <p style="color: #333; font-size: 32px; font-weight: bold; margin: 10px 0 0 0;">{actual_value:.2f}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.markdown(
                        f"""
                        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
                            <h3 style="color: #76B900; margin: 0; font-size: 18px;">Predicted Value</h3>
                            <p style="color: #333; font-size: 32px; font-weight: bold; margin: 10px 0 0 0;">{predicted_value:.2f}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            except Exception as e:
                st.warning(f"Could not generate LIME analysis: {e}")

        st.markdown("---")

        # Residual Analysis
        st.header("Residual Analysis")

        residuals = y_test_viz - y_pred_viz

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                x=residuals,
                title="Distribution of Residuals",
                labels={"x": "Residual", "y": "Count"},
                color_discrete_sequence=["#76B900"],
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(
                x=y_pred_viz,
                y=residuals,
                title="Residual Plot",
                labels={"x": "Predicted Values", "y": "Residuals"},
                color_discrete_sequence=["#76B900"],
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
