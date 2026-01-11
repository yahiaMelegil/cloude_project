# config.py - UPDATED FOR GOOGLE COLAB
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'json', 'txt', 'pdf', 'xlsx', 'xls'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

# Google Colab optimized Spark configuration
SPARK_CONFIG = {
    'app_name': 'SparkDataAnalytics',
    'master': 'local[*]',  # Use all available cores
    'executor_memory': '2g',  # Reduced for Colab
    'driver_memory': '2g',   # Reduced for Colab
    'executor_cores': 2,
    'spark.sql.execution.arrow.pyspark.enabled': 'true',
    'spark.ui.showConsoleProgress': 'false',
    'spark.driver.maxResultSize': '1g'
}

PERFORMANCE_CONFIGS = [
    {'name': 'Single_Machine_1Core', 'executors': 1, 'cores': 1, 'memory': '1g'},
    {'name': 'Single_Machine_2Core', 'executors': 1, 'cores': 2, 'memory': '2g'},
    {'name': 'Single_Machine_4Core', 'executors': 1, 'cores': 4, 'memory': '2g'}
]

ML_PARAMS = {
    'test_size': 0.2,
    'random_state': 42,
    'kmeans_k': 5,
    'regression_max_iter': 100,
    'rf_num_trees': 20,
    'rf_max_depth': 5
}

UI_CONFIG = {
    'page_title': 'Spark Data Analytics',
    'page_icon': '🚀',
    'layout': 'wide',
    'theme': 'light'
}

# Google Colab specific settings
if 'COLAB_GPU' in os.environ:
    SPARK_CONFIG['driver_memory'] = '1g'
    SPARK_CONFIG['executor_memory'] = '1g'
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB limit for Colab