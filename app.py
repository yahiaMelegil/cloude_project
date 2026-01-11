
import streamlit as st
import pandas as pd
import plotly.express as px
from pyspark.sql import SparkSession
import json
import sys
import time
from pathlib import Path
import os

sys.path.append(str(Path(__file__).parent))

# Custom Spark Session Manager to handle conflicts
class SparkSessionManager:
    """Manages Spark sessions to prevent conflicts in Google Colab"""
    
    _instance = None
    _current_spark = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SparkSessionManager, cls).__new__(cls)
        return cls._instance
    
    def stop_all_spark_sessions(self):
        """Stop all existing Spark sessions"""
        try:
            # Try to get and stop existing SparkContext
            from pyspark import SparkContext
            sc = SparkContext.getOrCreate()
            sc.stop()
            time.sleep(1)
        except:
            pass
        
        try:
            # Try to get and stop existing SparkSession
            spark = SparkSession.builder.getOrCreate()
            spark.stop()
            time.sleep(1)
        except:
            pass
        
        # Clear any cached sessions
        SparkSession._instantiatedSession = None
        self._current_spark = None
    
    def get_session(self, config=None):
        """Get a Spark session with proper cleanup"""
        # First cleanup any existing sessions
        self.stop_all_spark_sessions()
        time.sleep(1)
        
        if config is None:
            config = {
                'app_name': 'SparkDataAnalytics',
                'master': 'local[*]',
                'executor_memory': '2g',
                'driver_memory': '2g',
                'executor_cores': 2
            }
        
        # Build Spark session
        builder = SparkSession.builder \
            .appName(config['app_name']) \
            .master(config['master']) \
            .config("spark.driver.memory", config['driver_memory']) \
            .config("spark.executor.memory", config['executor_memory']) \
            .config("spark.ui.showConsoleProgress", "false") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.driver.maxResultSize", "1g") \
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
        
        spark = builder.getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        
        self._current_spark = spark
        return spark
    
    def stop_session(self):
        """Stop current session"""
        if self._current_spark is not None:
            try:
                self._current_spark.stop()
                self._current_spark = None
            except:
                pass

# Initialize the manager
spark_manager = SparkSessionManager()

# Import your modules AFTER SparkSessionManager is defined
from utils.config import SPARK_CONFIG
from utils.data_validator import DataValidator
from utils.file_handler import FileHandler
from spark_jobs.statistics import DataStatistics
from spark_jobs.ml_models import MLModels

# Page Configuration
st.set_page_config(
    page_title="Spark Data Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern design
st.markdown("""
<style>
    /* Modern gradient background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin-bottom: 24px;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Section headers */
    .section-header {
        color: #1e40af;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 3px solid #3b82f6;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 32px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
    }
    
    /* Progress steps */
    .step-container {
        display: flex;
        justify-content: space-between;
        margin: 32px 0;
        position: relative;
    }
    
    .step {
        display: flex;
        flex-direction: column;
        align-items: center;
        z-index: 2;
    }
    
    .step-number {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        background: #e5e7eb;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 18px;
        margin-bottom: 8px;
        transition: all 0.3s ease;
    }
    
    .step.active .step-number {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        color: white;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3);
    }
    
    .step.completed .step-number {
        background: #10b981;
        color: white;
    }
    
    .step-label {
        font-size: 14px;
        font-weight: 600;
        color: #6b7280;
    }
    
    .step.active .step-label {
        color: #1f2937;
    }
    
    .step-line {
        position: absolute;
        top: 24px;
        left: 10%;
        right: 10%;
        height: 4px;
        background: #e5e7eb;
        z-index: 1;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def show_progress_steps(current_step):
    """Display progress steps"""
    steps = [
        {"number": 1, "label": "Upload Data", "key": "upload"},
        {"number": 2, "label": "Preview", "key": "preview"},
        {"number": 3, "label": "Statistics", "key": "stats"},
        {"number": 4, "label": "ML Models", "key": "ml"},
        {"number": 5, "label": "Performance", "key": "perf"}
    ]
    
    # Find the index of current step
    current_step_index = next((i for i, step in enumerate(steps) if step["key"] == current_step), 0)
    
    html = """
    <div class="step-container">
        <div class="step-line"></div>
    """
    
    for step in steps:
        step_index = steps.index(step)
        
        # Determine step status
        if step["key"] == current_step:
            active = "active"
            completed = ""
        elif step_index < current_step_index:
            active = ""
            completed = "completed"
        else:
            active = ""
            completed = ""
        
        html += f"""
        <div class="step {active} {completed}">
            <div class="step-number">{step['number']}</div>
            <div class="step-label">{step['label']}</div>
        </div>
        """
    
    html += "</div>"
    
    # Render as HTML (not markdown)
    st.components.v1.html(f"""
    <style>
        .step-container {{
            display: flex;
            justify-content: space-between;
            margin: 32px 0;
            position: relative;
        }}
        
        .step {{
            display: flex;
            flex-direction: column;
            align-items: center;
            z-index: 2;
        }}
        
        .step-number {{
            width: 48px;
            height: 48px;
            border-radius: 50%;
            background: #e5e7eb;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 18px;
            margin-bottom: 8px;
            transition: all 0.3s ease;
        }}
        
        .step.active .step-number {{
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            color: white;
            box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3);
        }}
        
        .step.completed .step-number {{
            background: #10b981;
            color: white;
        }}
        
        .step-label {{
            font-size: 14px;
            font-weight: 600;
            color: #6b7280;
        }}
        
        .step.active .step-label {{
            color: #1f2937;
        }}
        
        .step-line {{
            position: absolute;
            top: 24px;
            left: 10%;
            right: 10%;
            height: 4px;
            background: #e5e7eb;
            z-index: 1;
        }}
    </style>
    {html}
    """, height=100)
def main():
    # Cleanup any existing Spark sessions at startup
    spark_manager.stop_all_spark_sessions()
    
    # Title and header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center; color: white; font-size: 42px; margin-bottom: 8px;'>🚀 Spark Data Analytics</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.9); font-size: 18px; margin-bottom: 32px;'>Cloud-Powered Data Processing & Machine Learning</p>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'upload'
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'stats_computed' not in st.session_state:
        st.session_state.stats_computed = False
    if 'spark_initialized' not in st.session_state:
        st.session_state.spark_initialized = False
    
    # Progress steps
    show_progress_steps(st.session_state.current_step)
    
    # Main content area
    main_container = st.container()
    
    with main_container:
        # Step 1: Upload Data
        if st.session_state.current_step == 'upload':
            render_upload_step()
        
        # Step 2: Preview Data
        elif st.session_state.current_step == 'preview':
            render_preview_step()
        
        # Step 3: Statistics
        elif st.session_state.current_step == 'stats':
            render_statistics_step()
        
        # Step 4: Machine Learning
        elif st.session_state.current_step == 'ml':
            render_ml_step()
        
        # Step 5: Performance Testing
        elif st.session_state.current_step == 'perf':
            render_performance_step()

def render_upload_step():
    """Step 1: File Upload"""
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        st.markdown("<h2 class='section-header'>📁 Upload Your Dataset</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color: #6b7280; margin-bottom: 24px;'>Supported formats: CSV, JSON, Excel (XLSX, XLS), Text</p>", unsafe_allow_html=True)
        
        # File upload with custom styling
        uploaded_file = st.file_uploader(
            "",
            type=['csv', 'json', 'xlsx', 'xls', 'txt'],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            validator = DataValidator()
            
            if not validator.validate_file_extension(uploaded_file.name):
                st.error("❌ Unsupported file format!")
                return
            
            if not validator.validate_file_size(uploaded_file):
                st.error("❌ File size exceeds limit (500MB)")
                return
            
            with st.spinner("Processing your file..."):
                file_handler = FileHandler()
                saved_path = file_handler.save_uploaded_file(uploaded_file)
                
                df_pandas, error = validator.load_data(saved_path)
                
                if error:
                    st.error(f"❌ {error}")
                    return
                
                is_valid, error_msg = validator.validate_data(df_pandas)
                if not is_valid:
                    st.error(f"❌ {error_msg}")
                    return
            
            # Store data in session state
            st.session_state.file_path = saved_path
            st.session_state.file_name = uploaded_file.name
            st.session_state.df_pandas = df_pandas
            st.session_state.file_uploaded = True
            
            st.success(f"✅ **{uploaded_file.name}** uploaded successfully!")
            
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("Next: Preview →", type="primary", use_container_width=True):
                    st.session_state.current_step = 'preview'
                    st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

def render_preview_step():
    """Step 2: Data Preview"""
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        st.markdown(f"<h2 class='section-header'>📋 Dataset Preview: {st.session_state.file_name}</h2>", unsafe_allow_html=True)
        
        # Show dataset info
        validator = DataValidator()
        data_info = validator.get_data_info(st.session_state.df_pandas)
        
        # Metrics in columns
        cols = st.columns(4)
        metrics = [
            ("📊 Rows", f"{data_info['num_rows']:,}"),
            ("🏗️ Columns", data_info['num_columns']),
            ("💾 Memory", f"{data_info['memory_usage']:.2f} MB"),
            ("✅ Missing", "No" if not data_info['has_nulls'] else "Yes")
        ]
        
        for col, (label, value) in zip(cols, metrics):
            col.metric(label, value)
        
        # Data preview with tabs
        tab1, tab2 = st.tabs(["First 10 Rows", "Data Types"])
        
        with tab1:
            st.dataframe(st.session_state.df_pandas.head(10), use_container_width=True)
        
        with tab2:
            dtype_df = pd.DataFrame({
                'Column': list(data_info['dtypes'].keys()),
                'Type': list(data_info['dtypes'].values())
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("← Back", use_container_width=True):
                st.session_state.current_step = 'upload'
                st.rerun()
        with col3:
            if st.button("Next: Statistics →", type="primary", use_container_width=True):
                st.session_state.current_step = 'stats'
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

def render_statistics_step():
    """Step 3: Statistics Analysis"""
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        st.markdown(f"<h2 class='section-header'>📊 Statistical Analysis</h2>", unsafe_allow_html=True)
        
        if st.button("🚀 Compute Statistics", type="primary", use_container_width=True):
            with st.spinner("Crunching numbers with Apache Spark..."):
                try:
                    # Get Spark session from manager
                    spark = spark_manager.get_session(SPARK_CONFIG)
                    
                    # Read data
                    df_spark = spark.read.csv(
                        st.session_state.file_path, 
                        header=True, 
                        inferSchema=True
                    )
                    
                    # Cache for better performance
                    df_spark.cache()
                    df_spark.count()  # Force caching
                    
                    # Compute statistics
                    stats = DataStatistics(spark)
                    results = stats.generate_full_report(df_spark)
                    
                    # Uncache to free memory
                    df_spark.unpersist()
                    
                    # Save results
                    FileHandler.save_results(results, Path(st.session_state.file_path).stem + "_statistics")
                    
                    st.session_state.last_stats = results
                    st.session_state.stats_computed = True
                    st.session_state.spark_initialized = True
                    
                    # Display results
                    display_statistics_results(results)
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}")
                    spark_manager.stop_all_spark_sessions()
                    return
        
        elif hasattr(st.session_state, 'last_stats'):
            display_statistics_results(st.session_state.last_stats)
        
        # Navigation buttons
        if st.session_state.get('stats_computed', False):
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("← Back to Preview", use_container_width=True):
                    st.session_state.current_step = 'preview'
                    st.rerun()
            with col3:
                if st.button("Next: ML Models →", type="primary", use_container_width=True):
                    st.session_state.current_step = 'ml'
                    st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

def display_statistics_results(results):
    """Display statistics results in a formatted way"""
    basic_stats = results['basic_statistics']
    
    # Dataset Overview
    st.markdown("### 📈 Dataset Overview")
    cols = st.columns(3)
    cols[0].metric("Total Rows", f"{basic_stats['dataset_size']['num_rows']:,}")
    cols[1].metric("Total Columns", basic_stats['dataset_size']['num_columns'])
    cols[2].metric("Processing Time", f"{basic_stats['execution_time_seconds']}s")
    
    # Data Types Visualization
    if basic_stats.get('data_types'):
        st.markdown("### 🔤 Data Types Distribution")
        dtypes_df = pd.DataFrame(list(basic_stats['data_types'].items()), columns=['Type', 'Count'])
        fig1 = px.pie(dtypes_df, values='Count', names='Type', 
                      hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig1, use_container_width=True)
    
    # Missing Values
    st.markdown("### ❓ Missing Values Analysis")
    missing_data = []
    for col, stats in basic_stats.get('missing_values', {}).items():
        if stats['null_count'] > 0:
            missing_data.append({
                'Column': col, 
                'Missing Count': stats['null_count'], 
                'Missing %': stats['null_percentage']
            })
    
    if missing_data:
        missing_df = pd.DataFrame(missing_data)
        st.dataframe(missing_df.sort_values('Missing %', ascending=False), use_container_width=True)
        fig2 = px.bar(missing_df, x='Column', y='Missing %', 
                     title='Missing Values by Column',
                     color='Missing %',
                     color_continuous_scale='Reds')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.success("🎉 Perfect! No missing values found in the dataset.")
    
    # Numeric Statistics
    if basic_stats.get('numeric_statistics'):
        st.markdown("### 🔢 Numeric Column Statistics")
        numeric_data = []
        for col, stats in basic_stats['numeric_statistics'].items():
            numeric_data.append({
                'Column': col, 
                'Min': stats.get('min'), 
                'Max': stats.get('max'),
                'Mean': stats.get('mean'), 
                'Std Dev': stats.get('std')
            })
        
        numeric_df = pd.DataFrame(numeric_data)
        st.dataframe(numeric_df, use_container_width=True)
        
        # Create visualization for means
        if len(numeric_df) > 0:
            fig3 = px.bar(numeric_df, x='Column', y='Mean', 
                         error_y=numeric_df['Std Dev'],
                         title='Mean Values with Standard Deviation',
                         color='Mean',
                         color_continuous_scale='Blues')
            st.plotly_chart(fig3, use_container_width=True)

def render_ml_step():
    """Step 4: Machine Learning Models"""
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        st.markdown(f"<h2 class='section-header'>🤖 Machine Learning Models</h2>", unsafe_allow_html=True)
        
        # Model selection
        model_type = st.selectbox(
            "Select Model Type:",
            ["Linear Regression", "Random Forest Regression", "K-Means Clustering", "FP-Growth", "Time Series"],
            key="model_select"
        )
        
        if st.session_state.get('spark_initialized'):
            try:
                # Get Spark session
                spark = spark_manager.get_session(SPARK_CONFIG)
                df_spark = spark.read.csv(
                    st.session_state.file_path, 
                    header=True, 
                    inferSchema=True
                )
                
                columns = df_spark.columns
                numeric_cols = [col for col, dtype in df_spark.dtypes 
                               if dtype in ['int', 'bigint', 'double', 'float']]
                
                st.success(f"✅ Data loaded: {df_spark.count()} rows, {len(columns)} columns")
                
                # Model-specific configuration
                if model_type == "Linear Regression":
                    col1, col2 = st.columns(2)
                    with col1:
                        label_col = st.selectbox("Target Variable (Y):", numeric_cols, key="lr_label")
                    with col2:
                        features = [c for c in numeric_cols if c != label_col]
                        feature_cols = st.multiselect(
                            "Features (X):", 
                            features, 
                            default=features[:min(3, len(features))],
                            key="lr_features"
                        )
                    
                    if st.button("🎯 Train Linear Regression", type="primary", use_container_width=True):
                        if not feature_cols:
                            st.error("Please select at least one feature!")
                            return
                        
                        with st.spinner("Training model..."):
                            try:
                                ml_models = MLModels(spark)
                                results = ml_models.linear_regression_model(
                                    df_spark, feature_cols, label_col
                                )
                                display_ml_results(results)
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                
                elif model_type == "K-Means Clustering":
                    col1, col2 = st.columns(2)
                    with col1:
                        feature_cols = st.multiselect(
                            "Features for Clustering:", 
                            numeric_cols, 
                            default=numeric_cols[:min(3, len(numeric_cols))],
                            key="kmeans_features"
                        )
                    with col2:
                        k = st.slider("Number of Clusters (k):", 2, 10, 5, key="kmeans_k")
                    
                    if st.button("🔮 Run K-Means Clustering", type="primary", use_container_width=True):
                        if not feature_cols:
                            st.error("Please select at least one feature!")
                            return
                        
                        with st.spinner(f"Clustering data into {k} groups..."):
                            try:
                                ml_models = MLModels(spark)
                                results = ml_models.kmeans_clustering(
                                    df_spark, feature_cols, k
                                )
                                display_clustering_results(results)
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
            
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                spark_manager.stop_all_spark_sessions()
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("← Back to Statistics", use_container_width=True):
                st.session_state.current_step = 'stats'
                st.rerun()
        with col3:
            if st.button("Next: Performance →", type="primary", use_container_width=True):
                st.session_state.current_step = 'perf'
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

def display_ml_results(results):
    """Display linear regression results"""
    st.markdown("### 📈 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{results['metrics']['r2']:.4f}")
    col2.metric("RMSE", f"{results['metrics']['rmse']:.4f}")
    col3.metric("MAE", f"{results['metrics']['mae']:.4f}")
    
    st.markdown(f"**Training Samples:** {results['training_samples']}")
    st.markdown(f"**Testing Samples:** {results['test_samples']}")
    
    # Coefficients
    st.markdown("### 🎯 Feature Coefficients")
    coef_df = pd.DataFrame({
        'Feature': results['feature_columns'],
        'Coefficient': results['coefficients']
    })
    st.dataframe(coef_df, use_container_width=True)
    
    fig = px.bar(coef_df, x='Feature', y='Coefficient', 
                 title='Feature Importance (Coefficients)',
                 color='Coefficient',
                 color_continuous_scale='RdYlBu_r')
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"⏱️ Training Time: {results['execution_time_seconds']} seconds")
    
    # Save results
    FileHandler.save_results(results, f"linear_regression_{results['label_column']}")

def display_clustering_results(results):
    """Display K-Means clustering results"""
    st.markdown("### 🎯 Clustering Results")
    
    st.metric("Silhouette Score", f"{results['metrics']['silhouette_score']:.4f}")
    
    # Cluster distribution
    st.markdown("### 📊 Cluster Distribution")
    cluster_df = pd.DataFrame(
        list(results['cluster_distribution'].items()), 
        columns=['Cluster', 'Count']
    )
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(cluster_df, use_container_width=True)
    with col2:
        fig = px.pie(cluster_df, values='Count', names='Cluster', 
                    title='Cluster Size Distribution',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"⏱️ Processing Time: {results['execution_time_seconds']} seconds")
    
    # Save results
    FileHandler.save_results(results, "kmeans_clustering")

def render_performance_step():
    """Step 5: Performance Testing"""
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        st.markdown(f"<h2 class='section-header'>⚡ Performance Testing</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <p style='color: #6b7280; margin-bottom: 24px;'>
        Test model performance with different Spark configurations to measure scalability and efficiency.
        </p>
        """, unsafe_allow_html=True)
        
        # Test configuration
        col1, col2 = st.columns(2)
        with col1:
            test_model = st.selectbox(
                "Select Model to Test:",
                ["Linear Regression", "K-Means Clustering"],
                key="perf_model"
            )
        
        # Load data if needed
        if st.session_state.get('spark_initialized'):
            try:
                spark = spark_manager.get_session(SPARK_CONFIG)
                df_spark = spark.read.csv(
                    st.session_state.file_path, 
                    header=True, 
                    inferSchema=True
                )
                
                numeric_cols = [col for col, dtype in df_spark.dtypes 
                               if dtype in ['int', 'bigint', 'double', 'float']]
                
                # Model-specific parameters
                if test_model == "Linear Regression":
                    label_col = st.selectbox("Target Variable:", numeric_cols, key="perf_lr_label")
                    feature_cols = st.multiselect(
                        "Features:", 
                        [c for c in numeric_cols if c != label_col],
                        key="perf_lr_features"
                    )
                else:
                    feature_cols = st.multiselect("Features:", numeric_cols, key="perf_kmeans_features")
                    k = st.slider("Clusters:", 2, 10, 5, key="perf_kmeans_k")
                
                if st.button("🚀 Start Performance Test", type="primary", use_container_width=True):
                    if not feature_cols:
                        st.error("Please select features!")
                        return
                    
                    st.warning("⏳ Performance testing may take several minutes...")
                    
                    # Test configurations
                    configs = [
                        {'name': 'Single Core', 'cores': 1},
                        {'name': 'Dual Core', 'cores': 2},
                        {'name': 'Quad Core', 'cores': 4},
                        {'name': 'Octa Core', 'cores': 8}
                    ]
                    
                    results = []
                    progress_bar = st.progress(0)
                    
                    for idx, config in enumerate(configs):
                        with st.spinner(f"Testing {config['name']}..."):
                            try:
                                # Stop existing sessions
                                spark_manager.stop_all_spark_sessions()
                                time.sleep(1)
                                
                                # Create new session with specific cores
                                spark_test = SparkSession.builder \
                                    .appName(f"PerfTest_{config['name']}") \
                                    .master(f"local[{config['cores']}]") \
                                    .config("spark.driver.memory", "2g") \
                                    .config("spark.executor.memory", "2g") \
                                    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                                    .getOrCreate()
                                
                                spark_test.sparkContext.setLogLevel("ERROR")
                                
                                # Load data
                                df_test = spark_test.read.csv(st.session_state.file_path, header=True, inferSchema=True)
                                ml = MLModels(spark_test)
                                
                                # Run model
                                start_time = time.time()
                                
                                if test_model == "Linear Regression":
                                    result = ml.linear_regression_model(df_test, feature_cols, label_col)
                                else:
                                    result = ml.kmeans_clustering(df_test, feature_cols, k)
                                
                                exec_time = time.time() - start_time
                                
                                results.append({
                                    'configuration': config['name'],
                                    'num_cores': config['cores'],
                                    'execution_time': exec_time,
                                    'result': result
                                })
                                
                                # Stop the test session
                                spark_test.stop()
                                
                            except Exception as e:
                                st.error(f"Error with {config['name']}: {str(e)}")
                                spark_manager.stop_all_spark_sessions()
                        
                        progress_bar.progress((idx + 1) / len(configs))
                    
                    if results:
                        display_performance_results(results)
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                spark_manager.stop_all_spark_sessions()
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("← Back to ML", use_container_width=True):
                st.session_state.current_step = 'ml'
                st.rerun()
        with col3:
            if st.button("🏁 Complete", type="secondary", use_container_width=True):
                # Cleanup Spark sessions
                spark_manager.stop_all_spark_sessions()
                st.success("✅ Analysis Complete!")
                st.balloons()
        
        st.markdown("</div>", unsafe_allow_html=True)

def display_performance_results(results):
    """Display performance testing results"""
    baseline = results[0]['execution_time']
    perf_data = []
    
    for r in results:
        speedup = baseline / r['execution_time']
        efficiency = (speedup / r['num_cores']) * 100
        perf_data.append({
            'Configuration': r['configuration'],
            'Cores': r['num_cores'],
            'Time (s)': round(r['execution_time'], 3),
            'Speedup': round(speedup, 3),
            'Efficiency (%)': round(efficiency, 2)
        })
    
    st.markdown("### 📊 Performance Metrics")
    perf_df = pd.DataFrame(perf_data)
    st.dataframe(perf_df.style.highlight_max(subset=['Speedup'], color='lightgreen'), 
                 use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.line(perf_df, x='Cores', y='Time (s)', 
                      title='Execution Time vs Cores',
                      markers=True,
                      line_shape='spline')
        fig1.update_traces(line=dict(color='#ef4444', width=3))
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.line(perf_df, x='Cores', y='Speedup', 
                      title='Speedup vs Cores',
                      markers=True,
                      line_shape='spline')
        fig2.add_shape(type="line", x0=0, y0=1, x1=8, y1=8, 
                      line=dict(color="Green", width=2, dash="dash"))
        fig2.update_traces(line=dict(color='#10b981', width=3))
        st.plotly_chart(fig2, use_container_width=True)
    
    # Efficiency chart
    fig3 = px.bar(perf_df, x='Configuration', y='Efficiency (%)',
                  title='Resource Efficiency',
                  color='Efficiency (%)',
                  color_continuous_scale='Viridis')
    st.plotly_chart(fig3, use_container_width=True)
    
    # Summary metrics
    avg_speedup = perf_df['Speedup'].mean()
    max_efficiency = perf_df['Efficiency (%)'].max()
    
    col1, col2 = st.columns(2)
    col1.metric("Average Speedup", f"{avg_speedup:.2f}x")
    col2.metric("Peak Efficiency", f"{max_efficiency:.1f}%")
    
    # Scalability assessment
    if perf_df['Speedup'].iloc[-1] / perf_df['Speedup'].iloc[0] > 6:
        st.success("🎯 Excellent scalability - Near linear speedup achieved!")
    elif perf_df['Speedup'].iloc[-1] / perf_df['Speedup'].iloc[0] > 4:
        st.info("👍 Good scalability - Significant performance gains")
    else:
        st.warning("⚠️ Moderate scalability - Consider optimizing your configuration")
    
    # Save results
    FileHandler.save_results({'performance_data': perf_data}, "performance_test")

if __name__ == "__main__":
    main()