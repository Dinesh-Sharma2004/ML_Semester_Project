import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

st.set_page_config(layout="wide", page_title="SHS-Opt Dashboard")

numerical_cols = [
    'prev_gpa', 'avg_sleep_hours', 'sleep_variability',
    'self_reported_study_hours', 'library_days_per_week',
    'avg_daily_online_hours', 'typical_connection_window_start',
    'stress_level', 'mood_score', 'avg_steps_per_day'
]
categorical_cols = [
    'department', 'year', 'cafeteria_veg_pref',
    'roommate_conflicts_reported'
]
raw_features = numerical_cols + categorical_cols

@st.cache_data
def load_default_data():
    df = pd.read_csv('dashboard_data.csv')
    X_train_processed_df = pd.read_csv('shap_train_data.csv')
    shap_values_class1 = np.load('shap_values_class1.npy')
    try:
        learning_curve_data = joblib.load('learning_curve_data.joblib')
    except FileNotFoundError:
        learning_curve_data = None
    return df, X_train_processed_df, shap_values_class1, learning_curve_data

@st.cache_resource
def load_models():
    try:
        classifier_pipeline = joblib.load('shs_classifier_pipeline.joblib')
        explainer = joblib.load('shs_shap_explainer.joblib')
        pca_model = joblib.load('shs_pca_model.joblib')
        preprocessor = classifier_pipeline.named_steps['preprocessor']
        feature_names = preprocessor.get_feature_names_out()
        model_name = classifier_pipeline.named_steps['model'].__class__.__name__
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error("Please run pipeline.py first to generate model artifacts.")
        return None, None, None, None, None, None
    return classifier_pipeline, explainer, preprocessor, feature_names, model_name, pca_model

def process_uploaded_file(uploaded_file, pipeline, preprocessor, pca):
    try:
        if uploaded_file.name.endswith('.csv'):
            df_upload = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df_upload = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

    missing_cols = [col for col in raw_features if col not in df_upload.columns]
    if missing_cols:
        st.error(f"Uploaded file is missing required columns: {', '.join(missing_cols)}")
        return None

    st.info("Processing uploaded data...")
    X_upload_raw = df_upload[raw_features]
    X_upload_processed = preprocessor.transform(X_upload_raw)
    X_upload_pca = pca.transform(X_upload_processed)

    df_upload['risk_probability'] = pipeline.predict_proba(X_upload_raw)[:, 1]
    df_upload['risk_prediction'] = pipeline.predict(X_upload_raw)
    df_upload['PC1'] = X_upload_pca[:, 0]
    df_upload['PC2'] = X_upload_pca[:, 1]
    df_upload['PC3'] = X_upload_pca[:, 2]
    st.success("Uploaded data processed successfully!")
    return df_upload

classifier_pipeline, explainer, preprocessor, processed_feature_names, model_name, pca_model = load_models()
if classifier_pipeline is None:
    st.stop()

X_train_processed_df, shap_values_class1, learning_curve_data = load_default_data()[1:]

st.sidebar.title("Data Source")
st.sidebar.info(f"**Active Model:** `{model_name}`")

uploaded_file = st.sidebar.file_uploader("Upload your own student data (CSV/Excel)", type=['csv', 'xlsx', 'xls'])

if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None

if uploaded_file is not None:
    if st.sidebar.button("Analyze Uploaded Data"):
        with st.spinner("Processing..."):
            st.session_state.processed_df = process_uploaded_file(
                uploaded_file, classifier_pipeline, preprocessor, pca_model
            )

if st.session_state.processed_df is not None:
    st.sidebar.success("✅ Using Uploaded Data")
    df = st.session_state.processed_df
    if st.sidebar.button("Clear Uploaded Data"):
        st.session_state.processed_df = None
        st.rerun()
else:
    st.sidebar.info("ℹ️ Using Default Synthetic Data")
    df = load_default_data()[0]

has_labels = 'at_risk_next_month' in df.columns
if has_labels:
    split_idx = int(len(df) * 0.8)
    df_test = df.iloc[split_idx:] if len(df) > 5 else df
    y_test_true = df_test['at_risk_next_month']
    y_test_pred_prob = df_test['risk_probability']
    y_test_pred = df_test['risk_prediction']

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Hostel Risk Overview",
    "Student Deep-Dive",
    "Live Prediction",
    "Model Performance",
    "Feature Importance"
])

if page == "Hostel Risk Overview":
    st.title("Hostel Wellbeing & Risk Overview")
    at_risk_students = df[df['risk_probability'] > 0.5].sort_values(by='risk_probability', ascending=False)
    st.header(f"Students Flagged for At-Risk Review ({len(at_risk_students)})")
    st.warning(f"**{len(at_risk_students)} students** currently have a >50% predicted risk of poor academic outcomes or high stress next month.")
    cols_to_show = ['student_id', 'risk_probability', 'prev_gpa', 'stress_level', 'avg_sleep_hours', 'self_reported_study_hours']
    cols_to_display = [col for col in cols_to_show if col in df.columns]
    st.dataframe(at_risk_students[cols_to_display].style.format({
        'risk_probability': '{:.1%}', 'prev_gpa': '{:.2f}', 'avg_sleep_hours': '{:.1f}', 'self_reported_study_hours': '{:.1f}'
    }))

elif page == "Student Deep-Dive":
    st.title("Student Deep-Dive & Intervention")
    if 'student_id' in df.columns:
        student_selector = st.selectbox("Select a Student", df['student_id'].unique())
        student_data_row = df[df['student_id'] == student_selector]
    else:
        student_selector = st.number_input("Select a Student (by row index)", 0, len(df)-1, 0)
        student_data_row = df.iloc[[student_selector]]
    if not student_data_row.empty:
        student_data = student_data_row.iloc[0]
        student_X_raw = student_data_row[raw_features]
        prob = student_data['risk_probability']
        st.header(f"Profile: {student_selector}")
        if prob > 0.5: st.error(f"**Predicted Risk: {prob:.1%} (HIGH)**")
        elif prob > 0.25: st.warning(f"**Predicted Risk: {prob:.1%} (MEDIUM)**")
        else: st.success(f"**Predicted Risk: {prob:.1%} (LOW)**")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Key Academic & Behavioral Vitals")
            st.metric("Previous GPA", f"{student_data['prev_gpa']:.2f}")
            st.metric("Self-Reported Stress (1-5)", f"{student_data['stress_level']:.1f}")
            st.metric("Avg. Sleep per Night", f"{student_data['avg_sleep_hours']:.1f} hrs")
        with col2:
            st.subheader("Study & Social Metrics")
            st.metric("Self-Reported Study", f"{student_data['self_reported_study_hours']:.1f} hrs/wk")
            st.metric("Library Visits", f"{student_data['library_days_per_week']:.0f} days/wk")
            st.metric("Avg. Online Time", f"{student_data['avg_daily_online_hours']:.1f} hrs/day")
        st.subheader("What's Driving This Student's Risk Score?")
        try:
            student_X_processed_np = preprocessor.transform(student_X_raw)
            student_X_processed_df = pd.DataFrame(student_X_processed_np, columns=processed_feature_names)
            explanation = explainer(student_X_processed_df)
            if len(explanation.values.shape) == 3 and explanation.values.shape[2] == 2:
                shap_values_class1 = explanation.values[0, :, 1]
                expected_value = explanation.base_values[0, 1]
            else:
                shap_values_class1 = explanation.values[0]
                expected_value = explanation.base_values[0]
            waterfall_explanation = shap.Explanation(values=shap_values_class1, base_values=expected_value, data=student_X_processed_df.iloc[0].values, feature_names=processed_feature_names)
            fig, ax = plt.subplots()
            shap.plots.waterfall(waterfall_explanation, max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(fig, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            st.error(f"Could not generate SHAP plot: {e}")

elif page == "Live Prediction":
    st.title("Live Risk Prediction")
    st.write("Enter a student's data manually to get a real-time risk prediction.")
    with st.form("prediction_form"):
        st.header("Student Input Data")
        col1, col2 = st.columns(2)
        with col1:
            prev_gpa = st.number_input("Previous GPA (0.0 - 4.0)", 0.0, 4.0, 3.0, 0.1)
            self_reported_study_hours = st.number_input("Self-Reported Study (hours/week)", 0, 70, 15)
            library_days_per_week = st.number_input("Library Visits (days/week)", 0, 7, 2)
            avg_daily_online_hours = st.number_input("Avg. WiFi/Online (hours/day)", 0.0, 24.0, 4.0, 0.5)
        with col2:
            stress_level = st.slider("Self-Reported Stress (1-5)", 1, 5, 3)
            mood_score = st.slider("Self-Reported Mood (1-5)", 1, 5, 3)
            avg_sleep_hours = st.number_input("Avg. Sleep (hours/night)", 0.0, 16.0, 7.0, 0.5)
            sleep_variability = st.number_input("Sleep Variability (Std. Dev.)", 0.0, 5.0, 1.0, 0.1)
        col3, col4, col5, col6 = st.columns(4)
        avg_steps_per_day = col3.number_input("Avg. Steps (per day)", 0, 30000, 8000, 500)
        department = col4.selectbox("Department", ('CS', 'ECE', 'MECH', 'CIVIL'))
        year = col5.selectbox("Year of Study", (1, 2, 3, 4))
        cafeteria_veg_pref = col6.selectbox("Cafeteria Preference", (0, 1), format_func=lambda x: "Veg" if x == 1 else "Non-Veg")
        typical_connection_window_start = 10
        roommate_conflicts_reported = 0
        submitted = st.form_submit_button("Predict Student Risk")
    if submitted:
        input_data = {'prev_gpa': [prev_gpa], 'avg_sleep_hours': [avg_sleep_hours], 'sleep_variability': [sleep_variability], 'self_reported_study_hours': [self_reported_study_hours], 'library_days_per_week': [library_days_per_week], 'avg_daily_online_hours': [avg_daily_online_hours], 'typical_connection_window_start': [typical_connection_window_start], 'stress_level': [stress_level], 'mood_score': [mood_score], 'avg_steps_per_day': [avg_steps_per_day], 'department': [department], 'year': [year], 'cafeteria_veg_pref': [cafeteria_veg_pref], 'roommate_conflicts_reported': [roommate_conflicts_reported]}
        input_df = pd.DataFrame(input_data)[raw_features]
        prob = classifier_pipeline.predict_proba(input_df)[0, 1]
        col1, col2 = st.columns([1, 2])
        with col1:
            if prob > 0.5: st.error(f"**Predicted Risk: {prob:.1%} (HIGH)**")
            elif prob > 0.25: st.warning(f"**Predicted Risk: {prob:.1%} (MEDIUM)**")
            else: st.success(f"**Predicted Risk: {prob:.1%} (LOW)**")
            st.metric("Risk Score", f"{prob:.1%}")
        with col2:
            try:
                student_X_processed_np = preprocessor.transform(input_df)
                student_X_processed_df = pd.DataFrame(student_X_processed_np, columns=processed_feature_names)
                explanation = explainer(student_X_processed_df)
                if len(explanation.values.shape) == 3 and explanation.values.shape[2] == 2:
                    shap_values_class1 = explanation.values[0, :, 1]
                    expected_value = explanation.base_values[0, 1]
                else:
                    shap_values_class1 = explanation.values[0]
                    expected_value = explanation.base_values[0]
                waterfall_explanation = shap.Explanation(values=shap_values_class1, base_values=expected_value, data=student_X_processed_df.iloc[0].values, feature_names=processed_feature_names)
                fig, ax = plt.subplots()
                shap.plots.waterfall(waterfall_explanation, max_display=10, show=False)
                plt.tight_layout()
                st.pyplot(fig, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                st.error(f"Could not generate SHAP plot: {e}")

elif page == "Model Performance":
    st.title("Model Performance")
    st.info(f"**Best Model Selected by Pipeline:** `{model_name}`")
    if has_labels:
        col1, col2 = st.columns(2)
        with col1:
            cm = confusion_matrix(y_test_true, y_test_pred)
            fig_cm = ff.create_annotated_heatmap(z=cm, x=['Predicted Not At-Risk', 'Predicted At-Risk'], y=['Actual Not At-Risk', 'Actual At-Risk'], colorscale='Blues', showscale=True)
            st.plotly_chart(fig_cm, use_container_width=True)
        with col2:
            fig_hist = px.histogram(df_test, x='risk_probability', color='at_risk_next_month', nbins=50, barmode='overlay', marginal='box', labels={'at_risk_next_month': 'True Label'})
            st.plotly_chart(fig_hist, use_container_width=True)
        col3, col4 = st.columns(2)
        with col3:
            fpr, tpr, _ = roc_curve(y_test_true, y_test_pred_prob)
            roc_auc = auc(fpr, tpr)
            fig_roc = px.area(x=fpr, y=tpr, title=f'ROC Curve (AUC = {roc_auc:.4f})', labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'})
            fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(fig_roc, use_container_width=True)
        with col4:
            prec, recall, _ = precision_recall_curve(y_test_true, y_test_pred_prob)
            pr_auc = auc(recall, prec)
            fig_pr = px.area(x=recall, y=prec, title=f'Precision-Recall Curve (AUC = {pr_auc:.4f})', labels={'x': 'Recall', 'y': 'Precision'})
            st.plotly_chart(fig_pr, use_container_width=True)
    else:
        st.warning("Uploaded data does not contain 'at_risk_next_month'.")
    st.subheader("Learning Curve")
    if learning_curve_data:
        lc_df = pd.DataFrame({'Training Samples': learning_curve_data['train_sizes'], 'Training Score': learning_curve_data['train_scores_mean'], 'Validation Score': learning_curve_data['test_scores_mean']})
        lc_df_melted = lc_df.melt('Training Samples', var_name='Score Type', value_name='Weighted F1-Score')
        fig_lc = px.line(lc_df_melted, x='Training Samples', y='Weighted F1-Score', color='Score Type', title='Model Learning Curve')
        fig_lc.update_yaxes(range=[0.0, 1.0])
        st.plotly_chart(fig_lc, use_container_width=True)
    else:
        st.warning("Learning curve data not found.")

elif page == "Feature Importance":
    st.title("Global & Local Feature Importance")
    st.image('shap_summary_beeswarm.png')
    st.subheader("Interactive SHAP Dependence Plot")
    col1, col2 = st.columns(2)
    with col1:
        feature_select = st.selectbox("Select feature (X-axis):", options=processed_feature_names, index=list(processed_feature_names).index('num__stress_level'))
    with col2:
        interaction_options = ["None"] + list(processed_feature_names)
        interaction_select = st.selectbox("Select interaction feature:", options=interaction_options, index=0)
    interaction_index = None if interaction_select == "None" else interaction_select
    fig, ax = plt.subplots()
    try:
        shap_values_class1_np = np.array(shap_values_class1)
        shap.dependence_plot(feature_select, shap_values_class1_np, X_train_processed_df, ax=ax, show=False, interaction_index=interaction_index)
        plt.tight_layout()
        st.pyplot(fig)
    except ValueError as ve:
        st.error(f"Could not generate dependence plot: {ve}")
    finally:
        plt.close(fig)
