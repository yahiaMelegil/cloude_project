# statistics.py - UPDATED
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, count, mean, stddev, min as spark_min, max as spark_max
from pyspark.sql.functions import sum as spark_sum, countDistinct, isnull, when, approx_count_distinct
from typing import Dict
import time

class DataStatistics:
    """Compute descriptive statistics using Spark"""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
    
    def compute_basic_stats(self, df: DataFrame) -> Dict:
        """
        Compute basic statistics optimized for Google Colab
        """
        start_time = time.time()
        
        results = {}
        
        try:
            # 1. Dataset Size
            num_rows = df.count()
            num_columns = len(df.columns)
            
            results['dataset_size'] = {
                'num_rows': num_rows,
                'num_columns': num_columns,
                'total_cells': num_rows * num_columns
            }
            
            # 2. Data Types Distribution
            dtype_counts = {}
            for col_name, dtype in df.dtypes:
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
            
            results['data_types'] = dtype_counts
            
            # 3. Missing Values Analysis (optimized for large datasets)
            missing_stats = {}
            # Sample first 1000 rows for missing values to avoid OOM
            sample_df = df.limit(1000) if num_rows > 1000 else df
            sample_count = sample_df.count()
            
            for col_name in df.columns[:20]:  # Limit to 20 columns
                null_count = sample_df.filter(col(col_name).isNull()).count()
                missing_pct = (null_count / sample_count) * 100 if sample_count > 0 else 0
                missing_stats[col_name] = {
                    'null_count': null_count,
                    'null_percentage': round(missing_pct, 2)
                }
            
            results['missing_values'] = missing_stats
            
            # 4. Numeric Columns Statistics (optimized)
            numeric_cols = [col_name for col_name, dtype in df.dtypes 
                           if dtype in ['int', 'bigint', 'double', 'float']]
            
            numeric_stats = {}
            # Limit to first 10 numeric columns for performance
            for col_name in numeric_cols[:10]:
                try:
                    stats_df = df.select(
                        spark_min(col(col_name)).alias('min'),
                        spark_max(col(col_name)).alias('max'),
                        mean(col(col_name)).alias('mean'),
                        stddev(col(col_name)).alias('std')
                    ).collect()[0]
                    
                    numeric_stats[col_name] = {
                        'min': float(stats_df['min']) if stats_df['min'] is not None else None,
                        'max': float(stats_df['max']) if stats_df['max'] is not None else None,
                        'mean': float(stats_df['mean']) if stats_df['mean'] is not None else None,
                        'std': float(stats_df['std']) if stats_df['std'] is not None else None
                    }
                except Exception as e:
                    numeric_stats[col_name] = {
                        'error': str(e),
                        'min': None,
                        'max': None,
                        'mean': None,
                        'std': None
                    }
            
            results['numeric_statistics'] = numeric_stats
            
        except Exception as e:
            results['error'] = f"Error computing statistics: {str(e)}"
        
        # Execution time
        execution_time = time.time() - start_time
        results['execution_time_seconds'] = round(execution_time, 3)
        
        return results
    
    def compute_unique_values(self, df: DataFrame, max_cols: int = 10) -> Dict:
        """
        Compute unique value counts for each column (optimized)
        """
        start_time = time.time()
        
        unique_counts = {}
        columns_to_check = df.columns[:max_cols]  # Limit to avoid long processing
        
        for col_name in columns_to_check:
            try:
                # Use approximate count for better performance
                if df.count() > 10000:
                    unique_count = df.select(approx_count_distinct(col_name)).collect()[0][0]
                else:
                    unique_count = df.select(col_name).distinct().count()
                
                total_count = df.count()
                uniqueness_ratio = round(unique_count / total_count, 4) if total_count > 0 else 0
                
                unique_counts[col_name] = {
                    'unique_values': unique_count,
                    'uniqueness_ratio': uniqueness_ratio
                }
            except Exception as e:
                unique_counts[col_name] = {
                    'unique_values': None,
                    'uniqueness_ratio': None,
                    'error': str(e)
                }
        
        execution_time = time.time() - start_time
        
        return {
            'unique_value_counts': unique_counts,
            'execution_time_seconds': round(execution_time, 3)
        }
    
    def compute_column_distributions(self, df: DataFrame, categorical_cols: list = None) -> Dict:
        """
        Compute value distributions for categorical columns
        """
        if categorical_cols is None:
            # Auto-detect categorical columns (string type with reasonable unique count)
            categorical_cols = [col_name for col_name, dtype in df.dtypes 
                              if dtype == 'string']
            categorical_cols = categorical_cols[:3]  # Limit to 3 columns for performance
        
        distributions = {}
        
        for col_name in categorical_cols:
            try:
                # Get top 10 most frequent values
                value_counts = df.groupBy(col_name).count() \
                                .orderBy(col('count').desc()) \
                                .limit(10) \
                                .collect()
                
                distributions[col_name] = [
                    {'value': row[col_name], 'count': row['count']} 
                    for row in value_counts
                ]
            except Exception as e:
                distributions[col_name] = [{'error': str(e)}]
        
        return {'distributions': distributions}
    
    def compute_correlations(self, df: DataFrame, numeric_cols: list = None) -> Dict:
        """
        Compute correlation matrix for numeric columns
        """
        if numeric_cols is None:
            numeric_cols = [col_name for col_name, dtype in df.dtypes 
                           if dtype in ['int', 'bigint', 'double', 'float']]
        
        if len(numeric_cols) < 2:
            return {'correlations': {}, 'message': 'Need at least 2 numeric columns'}
        
        # Limit to first 5 numeric columns for performance
        numeric_cols = numeric_cols[:5]
        
        correlations = {}
        try:
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    corr_value = df.stat.corr(col1, col2)
                    correlations[f"{col1}_vs_{col2}"] = round(corr_value, 4) if corr_value else None
        except Exception as e:
            correlations['error'] = str(e)
        
        return {'correlations': correlations}
    
    def generate_full_report(self, df: DataFrame) -> Dict:
        """
        Generate complete statistical report optimized for Google Colab
        """
        print("Computing statistics...")
        
        try:
            # Cache the dataframe for better performance
            df.cache()
            df.count()  # Force caching
            
            report = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'basic_statistics': self.compute_basic_stats(df),
                'unique_values': self.compute_unique_values(df),
                'correlations': self.compute_correlations(df)
            }
            
            # Uncache to free memory
            df.unpersist()
            
            return report
            
        except Exception as e:
            return {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'error': f"Error generating report: {str(e)}"
            }