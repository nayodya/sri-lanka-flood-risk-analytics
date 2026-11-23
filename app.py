
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Sri Lanka Flood Risk Analytics",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# LOAD DATASET AND MODELS
# --------------------------------------------------

@st.cache_data
def load_dataset():
    return pd.read_csv(
        "dataset/sri_lanka_flood_risk_modeled.csv"
    )

@st.cache_resource
def load_models():

    preprocessor = joblib.load(
        "models/preprocessor.pkl"
    )

    label_encoder = joblib.load(
        "models/label_encoder.pkl"
    )

    models = {
        "Decision Tree": joblib.load(
            "models/decision_tree.pkl"
        ),

        "Random Forest": joblib.load(
            "models/random_forest.pkl"
        ),

        "Extra Trees": joblib.load(
            "models/extra_trees.pkl"
        ),

        "XGBoost": joblib.load(
            "models/xgboost.pkl"
        ),

        "Stacking Classifier": joblib.load(
            "models/stacking_model.pkl"
        )
    }

    return preprocessor, label_encoder, models


@st.cache_data
def load_model_comparison():
    return pd.read_csv(
        "models/model_comparison.csv"
    )


@st.cache_data
def load_feature_importance():

    rf = pd.read_csv(
        "models/random_forest_feature_importance.csv"
    )

    xgb = pd.read_csv(
        "models/xgboost_feature_importance.csv"
    )

    permutation = pd.read_csv(
        "models/permutation_feature_importance.csv"
    )

    return rf, xgb, permutation


# Load everything
try:
    df = load_dataset()

    preprocessor, label_encoder, models = load_models()

    comparison_df = load_model_comparison()

    rf_importance, xgb_importance, permutation_importance = (
        load_feature_importance()
    )

except Exception as e:
    st.error(f"Error loading project files: {e}")
    st.stop()



# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main-header {
    font-size: 32px;
    font-weight: 700;
    color: #2563EB;
}

.sub-header {
    color: #64748B;
    font-size: 15px;
}

.metric-card {
    background-color: #F1F5F9;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: #1E293B;
}

.metric-label {
    color: #64748B;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🌧️ Flood Analytics")

st.sidebar.caption(
    "Sri Lanka Flood Risk Prediction System"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Dataset Analysis",
        "Model Comparison",
        "Feature Importance",
        "Flood Risk Prediction"
    ]
)

st.sidebar.divider()


st.sidebar.caption(
    "Ensemble Learning: Bagging and Stacking"
)


# --------------------------------------------------
# OVERVIEW PAGE
# --------------------------------------------------

if page == "Overview":

    st.markdown(
        '<div class="main-header">'
        'Sri Lanka Flood Risk Analytics'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-header">'
        'Ensemble Learning Based Flood Risk Prediction System'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # Dataset statistics
    total_records = len(df)

    total_districts = df["district"].nunique()

    total_models = len(models)

    best_row = comparison_df.sort_values(
        "Macro F1",
        ascending=False
    ).iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Records",
        f"{total_records:,}"
    )

    col2.metric(
        "Sri Lankan Districts",
        total_districts
    )

    col3.metric(
        "ML Models",
        total_models
    )

    col4.metric(
        "Best Model",
        best_row["Model"]
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Flood Risk Distribution")

        category_counts = df[
            "flood_category"
        ].value_counts()

        fig, ax = plt.subplots(figsize=(8, 5))

        category_counts.plot(
            kind="bar",
            ax=ax
        )

        ax.set_xlabel("Flood Category")
        ax.set_ylabel("Number of Records")
        ax.tick_params(axis="x", rotation=25)

        st.pyplot(fig)
        plt.close(fig)

    with right:

        st.subheader("Model Performance")

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.barplot(
            data=comparison_df.sort_values(
                "Macro F1",
                ascending=False
            ),
            x="Macro F1",
            y="Model",
            ax=ax
        )

        ax.set_xlim(0, 1)

        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    st.subheader("Project Description")

    st.write("""
    This application uses ensemble machine learning algorithms
    to analyze Sri Lankan flood-risk data.

    The implemented models include Decision Tree, Random Forest,
    Extra Trees, XGBoost and Stacking Classifier.

    The system provides dataset analysis, model performance
    comparison, feature importance analysis and interactive
    flood-risk predictions.
    """)


# --------------------------------------------------
# DATASET ANALYSIS PAGE
# --------------------------------------------------

elif page == "Dataset Analysis":

    st.title("📊 Dataset Analysis")

    st.write(
        "Explore flood-risk records across Sri Lankan districts."
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    st.divider()

    # District filter
    districts = ["All Districts"] + sorted(
        df["district"].dropna().unique().tolist()
    )

    selected_district = st.selectbox(
        "Select District",
        districts
    )

    if selected_district == "All Districts":
        filtered_df = df.copy()
    else:
        filtered_df = df[
            df["district"] == selected_district
        ]

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Flood Category Distribution")

        category_counts = filtered_df[
            "flood_category"
        ].value_counts()

        fig, ax = plt.subplots(figsize=(8, 5))

        category_counts.plot(
            kind="bar",
            ax=ax
        )

        ax.set_xlabel("Flood Category")
        ax.set_ylabel("Records")
        ax.tick_params(axis="x", rotation=25)

        st.pyplot(fig)
        plt.close(fig)

    with col2:

        st.subheader("Average Rainfall by Category")

        rainfall = filtered_df.groupby(
            "flood_category"
        )["precipitation_sum"].mean().sort_values(
            ascending=False
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        rainfall.plot(
            kind="bar",
            ax=ax
        )

        ax.set_ylabel("Average Rainfall (mm)")
        ax.set_xlabel("Flood Category")
        ax.tick_params(axis="x", rotation=25)

        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    st.subheader("District-wise Flood Records")

    district_counts = filtered_df[
        "district"
    ].value_counts().sort_values(ascending=False)

    st.bar_chart(district_counts)

    st.subheader("Rainfall Statistics")

    st.dataframe(
        filtered_df[
            ["precipitation_sum", "rain_48h", "rain_72h"]
        ].describe(),
        use_container_width=True
    )


# --------------------------------------------------
# MODEL COMPARISON PAGE
# --------------------------------------------------

elif page == "Model Comparison":

    st.title("🤖 Ensemble Model Comparison")

    st.write(
        "Compare the performance of five machine learning models."
    )

    st.subheader("Performance Comparison Table")

    display_df = comparison_df.sort_values(
        "Macro F1",
        ascending=False
    ).reset_index(drop=True)

    display_df.index = display_df.index + 1

    st.dataframe(
        display_df.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "Macro F1": "{:.4f}",
            "Balanced Accuracy": "{:.4f}",
            "Training Time (s)": "{:.2f}"
        }),
        use_container_width=True
    )

    st.divider()

    st.subheader("Performance Metrics")

    metric = st.selectbox(
        "Select Evaluation Metric",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "Macro F1",
            "Balanced Accuracy"
        ]
    )

    chart_df = display_df.sort_values(
        metric,
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=chart_df,
        x=metric,
        y="Model",
        ax=ax
    )

    ax.set_xlim(0, 1)

    ax.set_title(
        f"Model Comparison Based on {metric}"
    )

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader("Training Time Comparison")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=comparison_df.sort_values(
            "Training Time (s)",
            ascending=False
        ),
        x="Training Time (s)",
        y="Model",
        ax=ax
    )

    ax.set_title("Model Training Time")

    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "XGBoost currently has the highest Macro F1 score "
        "in the saved experiment. Random Forest has the "
        "highest accuracy, while Stacking has the highest "
        "balanced accuracy."
    )


# --------------------------------------------------
# FEATURE IMPORTANCE PAGE
# --------------------------------------------------

elif page == "Feature Importance":

    st.title("📈 Feature Importance Analysis")

    st.write(
        "Understand which variables contribute to flood-risk predictions."
    )

    feature_method = st.selectbox(
        "Select Analysis Method",
        [
            "Random Forest",
            "XGBoost",
            "Permutation Importance"
        ]
    )

    if feature_method == "Random Forest":

        importance_df = rf_importance

        importance_column = "Importance"

    elif feature_method == "XGBoost":

        importance_df = xgb_importance

        importance_column = "Importance"

    else:

        importance_df = permutation_importance

        importance_column = "Importance"

    top_n = st.slider(
        "Number of Features",
        min_value=5,
        max_value=20,
        value=15
    )

    top_features = importance_df.sort_values(
        importance_column,
        ascending=False
    ).head(top_n)

    fig, ax = plt.subplots(figsize=(11, 7))

    sns.barplot(
        data=top_features,
        x=importance_column,
        y="Feature",
        ax=ax
    )

    ax.set_title(
        f"{feature_method} - Top {top_n} Features"
    )

    ax.set_xlabel("Importance Score")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Feature Importance Data")

    st.dataframe(
        top_features,
        use_container_width=True
    )

    st.caption(
        "Feature importance indicates model reliance or "
        "performance sensitivity, not causal influence."
    )


# --------------------------------------------------
# FLOOD RISK PREDICTION PAGE
# --------------------------------------------------

elif page == "Flood Risk Prediction":

    st.title("🌧️ Flood Risk Prediction")

    st.write(
        "Enter weather and soil conditions to estimate "
        "the flood-risk category for a Sri Lankan district."
    )

    st.warning(
        "This is an experimental prediction based on modeled "
        "historical dataset labels. It is not an official flood "
        "warning or emergency alert."
    )

    st.divider()

    # Select model
    selected_model_name = st.selectbox(
        "Select Machine Learning Model",
        [
            "XGBoost",
            "Random Forest",
            "Decision Tree",
            "Extra Trees",
            "Stacking Classifier"
        ]
    )

    # District selection
    district_list = sorted(
        df["district"].dropna().unique().tolist()
    )

    selected_district = st.selectbox(
        "Select Sri Lankan District",
        district_list
    )

    # Obtain geographic information from dataset
    district_data = df[
        df["district"] == selected_district
    ]

    district_info = district_data.iloc[0]

    province = district_info["province"]

    climatic_zone = district_info["climatic_zone"]

    latitude = float(
        district_data["latitude"].mean()
    )

    longitude = float(
        district_data["longitude"].mean()
    )

    col1, col2, col3 = st.columns(3)

    col1.info(f"Province: {province}")

    col2.info(f"Climate Zone: {climatic_zone}")

    col3.info(
        f"Location: {latitude:.3f}, {longitude:.3f}"
    )

    st.divider()

    st.subheader("Weather Conditions")

    # Default values based on selected district
    def district_median(column):
        return float(
            district_data[column].median()
        )

    col1, col2 = st.columns(2)

    with col1:

        precipitation = st.number_input(
            "Daily Precipitation (mm)",
            min_value=0.0,
            value=district_median("precipitation_sum"),
            step=1.0
        )

        rain_48h = st.number_input(
            "48-Hour Rainfall (mm)",
            min_value=0.0,
            value=district_median("rain_48h"),
            step=1.0
        )

        rain_72h = st.number_input(
            "72-Hour Rainfall (mm)",
            min_value=0.0,
            value=district_median("rain_72h"),
            step=1.0
        )

        temperature = st.number_input(
            "Maximum Temperature (°C)",
            value=district_median("temperature_2m_max"),
            step=0.5
        )

    with col2:

        soil_moisture_0 = st.number_input(
            "Soil Moisture (0–7 cm)",
            min_value=0.0,
            value=district_median(
                "soil_moisture_0_to_7cm_mean"
            ),
            step=0.01
        )

        soil_moisture_7 = st.number_input(
            "Soil Moisture (7–28 cm)",
            min_value=0.0,
            value=district_median(
                "soil_moisture_7_to_28cm_mean"
            ),
            step=0.01
        )

        soil_saturation = st.number_input(
            "Soil Saturation Index",
            value=district_median(
                "soil_saturation_index"
            ),
            step=0.01
        )

        wind_speed = st.number_input(
            "Maximum Wind Speed (km/h)",
            min_value=0.0,
            value=district_median(
                "wind_speed_10m_max"
            ),
            step=0.5
        )

    st.divider()

    # Prediction button
    if st.button(
        "🔍 Predict Flood Risk",
        type="primary",
        use_container_width=True
    ):

        # Prepare input using the original training columns
        input_data = pd.DataFrame([{
            "precipitation_sum": precipitation,
            "soil_moisture_0_to_7cm_mean": soil_moisture_0,
            "soil_moisture_7_to_28cm_mean": soil_moisture_7,
            "temperature_2m_max": temperature,
            "wind_speed_10m_max": wind_speed,
            "latitude": latitude,
            "longitude": longitude,
            "rain_48h": rain_48h,
            "rain_72h": rain_72h,
            "soil_saturation_index": soil_saturation,
            "district": selected_district,
            "province": province,
            "climatic_zone": climatic_zone
        }])

        try:

            # Transform input
            processed_input = preprocessor.transform(
                input_data
            )

            # Get selected model
            selected_model = models[
                selected_model_name
            ]

            # Predict
            prediction = selected_model.predict(
                processed_input
            )

            # Decode encoded predictions where necessary
            if selected_model_name in [
                "XGBoost",
                "Stacking Classifier"
            ]:

                predicted_category = (
                    label_encoder.inverse_transform(
                        prediction.astype(int)
                    )[0]
                )

                probability_labels = (
                    label_encoder.classes_
                )

            else:

                predicted_category = prediction[0]

                probability_labels = (
                    selected_model.classes_
                )

            # Get probabilities
            probabilities = selected_model.predict_proba(
                processed_input
            )[0]

            # Display result
            st.divider()

            st.subheader("Prediction Result")

            st.success(
                f"Predicted Flood Category: {predicted_category}"
            )

            # Display probabilities
            probability_df = pd.DataFrame({
                "Flood Category": probability_labels,
                "Probability": probabilities
            })

            probability_df = probability_df.sort_values(
                "Probability",
                ascending=False
            )

            st.subheader("Prediction Probabilities")

            st.dataframe(
                probability_df.style.format({
                    "Probability": "{:.2%}"
                }),
                use_container_width=True
            )

            st.bar_chart(
                probability_df.set_index(
                    "Flood Category"
                )["Probability"]
            )

            st.caption(
                "Probabilities represent the selected model's "
                "estimated class probabilities, not calibrated "
                "real-world flood probabilities."
            )

        except Exception as e:

            st.error(
                f"Prediction error: {str(e)}"
            )

