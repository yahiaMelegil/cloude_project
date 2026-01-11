# ml_models.py - UPDATED FOR GOOGLE COLAB
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.clustering import KMeans
from pyspark.ml.fpm import FPGrowth
from pyspark.ml.evaluation import RegressionEvaluator, ClusteringEvaluator
from pyspark.sql.functions import col, year, month, dayofmonth, sum as spark_sum, avg, count
from pyspark.sql.functions import collect_list, array, lit
import time
from typing import Dict, List, Tuple


class MLModels:
    """Machine Learning models using Spark MLlib - Optimized for Google Colab"""

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def prepare_features(self, df: DataFrame, feature_cols: List[str],
                         label_col: str = None) -> Tuple[DataFrame, VectorAssembler]:
        """
        Prepare features for ML models - Optimized for Google Colab
        """
        # Handle categorical columns if any
        string_cols = [col_name for col_name, dtype in df.dtypes
                       if dtype == 'string' and col_name in feature_cols]

        for col_name in string_cols[:3]:  # Limit to 3 categorical columns
            try:
                indexer = StringIndexer(
                    inputCol=col_name, outputCol=f"{col_name}_indexed")
                df = indexer.fit(df).transform(df)
                feature_cols = [f"{col_name}_indexed" if c == col_name else c
                                for c in feature_cols]
            except Exception as e:
                print(f"Warning: Could not index column {col_name}: {str(e)}")
                # Remove problematic column
                feature_cols = [c for c in feature_cols if c != col_name]

        # Assemble features
        assembler = VectorAssembler(
            inputCols=feature_cols, outputCol="features")
        df = assembler.transform(df)

        return df, assembler

    def linear_regression_model(self, df: DataFrame, feature_cols: List[str],
                                label_col: str) -> Dict:
        """
        ML Model 1: Linear Regression - Optimized for Colab
        """
        start_time = time.time()
        print("Training Linear Regression Model...")

        try:
            # Prepare data
            df_prepared, assembler = self.prepare_features(
                df, feature_cols, label_col)

            # Cache data for better performance
            df_prepared.cache()
            df_prepared.count()  # Force caching

            # Split data
            train_data, test_data = df_prepared.randomSplit(
                [0.8, 0.2], seed=42)

            # Train model with optimized parameters for Colab
            lr = LinearRegression(
                featuresCol="features",
                labelCol=label_col,
                maxIter=50,  # Reduced for Colab
                regParam=0.01,
                elasticNetParam=0.8
            )
            model = lr.fit(train_data)

            # Make predictions
            predictions = model.transform(test_data)

            # Evaluate
            evaluator = RegressionEvaluator(
                labelCol=label_col, predictionCol="prediction")
            rmse = evaluator.evaluate(
                predictions, {evaluator.metricName: "rmse"})
            r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
            mae = evaluator.evaluate(
                predictions, {evaluator.metricName: "mae"})

            execution_time = time.time() - start_time

            results = {
                'model_type': 'Linear Regression',
                'feature_columns': feature_cols,
                'label_column': label_col,
                'metrics': {
                    'rmse': float(rmse) if rmse else None,
                    'r2': float(r2) if r2 else None,
                    'mae': float(mae) if mae else None
                },
                'coefficients': model.coefficients.toArray().tolist() if model.coefficients else [],
                'intercept': float(model.intercept) if model.intercept else None,
                'training_samples': train_data.count(),
                'test_samples': test_data.count(),
                'execution_time_seconds': round(execution_time, 3)
            }

            # Cleanup
            df_prepared.unpersist()

            return results

        except Exception as e:
            return {
                'model_type': 'Linear Regression',
                'error': str(e),
                'execution_time_seconds': round(time.time() - start_time, 3)
            }

    def random_forest_regression(self, df: DataFrame, feature_cols: List[str],
                                 label_col: str) -> Dict:
        """
        ML Model 2: Random Forest Regression - Optimized for Colab
        """
        start_time = time.time()
        print("Training Random Forest Regression Model...")

        try:
            # Prepare data
            df_prepared, assembler = self.prepare_features(
                df, feature_cols, label_col)

            # Cache data
            df_prepared.cache()
            df_prepared.count()

            # Split data
            train_data, test_data = df_prepared.randomSplit(
                [0.8, 0.2], seed=42)

            # Train model with reduced parameters for Colab
            rf = RandomForestRegressor(
                featuresCol="features",
                labelCol=label_col,
                numTrees=10,  # Reduced for Colab
                maxDepth=4,   # Reduced for Colab
                maxBins=32,
                seed=42
            )
            model = rf.fit(train_data)

            # Make predictions
            predictions = model.transform(test_data)

            # Evaluate
            evaluator = RegressionEvaluator(
                labelCol=label_col, predictionCol="prediction")
            rmse = evaluator.evaluate(
                predictions, {evaluator.metricName: "rmse"})
            r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

            execution_time = time.time() - start_time

            results = {
                'model_type': 'Random Forest Regression',
                'feature_columns': feature_cols,
                'label_column': label_col,
                'metrics': {
                    'rmse': float(rmse) if rmse else None,
                    'r2': float(r2) if r2 else None
                },
                'num_trees': 10,
                'feature_importances': model.featureImportances.toArray().tolist() if model.featureImportances else [],
                'training_samples': train_data.count(),
                'test_samples': test_data.count(),
                'execution_time_seconds': round(execution_time, 3)
            }

            # Cleanup
            df_prepared.unpersist()

            return results

        except Exception as e:
            return {
                'model_type': 'Random Forest Regression',
                'error': str(e),
                'execution_time_seconds': round(time.time() - start_time, 3)
            }

    def kmeans_clustering(self, df: DataFrame, feature_cols: List[str],
                          k: int = 5) -> Dict:
        """
        ML Model 3: K-Means Clustering - Optimized for Colab
        """
        start_time = time.time()
        print(f"Training K-Means Clustering Model (k={k})...")

        try:
            # Prepare data
            df_prepared, assembler = self.prepare_features(df, feature_cols)

            # Cache data
            df_prepared.cache()
            df_prepared.count()

            # Standardize features
            scaler = StandardScaler(
                inputCol="features", outputCol="scaled_features")
            scaler_model = scaler.fit(df_prepared)
            df_scaled = scaler_model.transform(df_prepared)

            # Train model with reduced iterations for Colab
            kmeans = KMeans(
                featuresCol="scaled_features",
                predictionCol="cluster",
                k=k,
                seed=42,
                maxIter=20,  # Reduced for Colab
                tol=1e-4
            )
            model = kmeans.fit(df_scaled)

            # Make predictions
            predictions = model.transform(df_scaled)

            # Evaluate
            evaluator = ClusteringEvaluator(
                featuresCol="scaled_features",
                predictionCol="cluster",
                metricName="silhouette"
            )
            silhouette = evaluator.evaluate(predictions)

            # Cluster sizes
            cluster_sizes = predictions.groupBy("cluster").count().collect()
            cluster_distribution = {row['cluster']
                : row['count'] for row in cluster_sizes}

            execution_time = time.time() - start_time

            results = {
                'model_type': 'K-Means Clustering',
                'feature_columns': feature_cols,
                'num_clusters': k,
                'metrics': {
                    'silhouette_score': float(silhouette) if silhouette else None
                },
                'cluster_centers': [center.toArray().tolist() for center in model.clusterCenters()],
                'cluster_distribution': cluster_distribution,
                'total_samples': df_prepared.count(),
                'execution_time_seconds': round(execution_time, 3)
            }

            # Cleanup
            df_prepared.unpersist()

            return results

        except Exception as e:
            return {
                'model_type': 'K-Means Clustering',
                'error': str(e),
                'execution_time_seconds': round(time.time() - start_time, 3)
            }

    def association_rules(self, df: DataFrame, transaction_col: str,
                          min_support: float = 0.05, min_confidence: float = 0.3) -> Dict:
        """
        ML Model 4: FP-Growth (Association Rules) - Optimized for Colab
        """
        start_time = time.time()
        print("Running FP-Growth for Association Rules...")

        try:
            # Prepare data - convert to list of items per transaction
            if isinstance(df.schema[transaction_col].dataType, type(df.schema['string'])):
                from pyspark.sql.functions import split
                df_prepared = df.withColumn(
                    "items", split(col(transaction_col), ","))
            else:
                df_prepared = df.withColumnRenamed(transaction_col, "items")

            # Cache data
            df_prepared.cache()
            df_prepared.count()

            # Train FP-Growth model with higher minSupport for Colab
            fpGrowth = FPGrowth(
                itemsCol="items",
                minSupport=min_support,  # Increased for Colab
                minConfidence=min_confidence,
                numPartitions=2  # Reduced for Colab
            )
            model = fpGrowth.fit(df_prepared)

            # Get frequent itemsets (limit to top 10 for Colab)
            frequent_itemsets = model.freqItemsets.collect()
            freq_items_list = [
                {'items': row['items'], 'freq': row['freq']}
                for row in frequent_itemsets[:10]  # Top 10 for Colab
            ]

            # Get association rules (limit to top 10 for Colab)
            association_rules = model.associationRules.collect()
            rules_list = [
                {
                    'antecedent': row['antecedent'],
                    'consequent': row['consequent'],
                    'confidence': float(row['confidence']),
                    'lift': float(row['lift'])
                }
                for row in association_rules[:10]  # Top 10 for Colab
            ]

            execution_time = time.time() - start_time

            results = {
                'model_type': 'FP-Growth Association Rules',
                'min_support': min_support,
                'min_confidence': min_confidence,
                'num_frequent_itemsets': len(frequent_itemsets),
                'num_association_rules': len(association_rules),
                'top_frequent_itemsets': freq_items_list,
                'top_association_rules': rules_list,
                'total_transactions': df_prepared.count(),
                'execution_time_seconds': round(execution_time, 3)
            }

            # Cleanup
            df_prepared.unpersist()

            return results

        except Exception as e:
            return {
                'model_type': 'FP-Growth Association Rules',
                'error': str(e),
                'execution_time_seconds': round(time.time() - start_time, 3)
            }

    def time_series_aggregation(self, df: DataFrame, date_col: str,
                                value_col: str, aggregation: str = 'monthly') -> Dict:
        """
        ML Model 5 (Bonus): Time Series Aggregation - Optimized for Colab
        """
        start_time = time.time()
        print(f"Performing {aggregation} time series aggregation...")

        try:
            # Ensure date column is date type
            from pyspark.sql.functions import to_date
            df_prepared = df.withColumn(date_col, to_date(col(date_col)))

            # Add time components
            df_prepared = df_prepared.withColumn("year", year(col(date_col)))
            df_prepared = df_prepared.withColumn("month", month(col(date_col)))
            df_prepared = df_prepared.withColumn(
                "day", dayofmonth(col(date_col)))

            # Cache data
            df_prepared.cache()
            df_prepared.count()

            # Aggregate based on period
            if aggregation == 'daily':
                grouped = df_prepared.groupBy("year", "month", "day")
            elif aggregation == 'monthly':
                grouped = df_prepared.groupBy("year", "month")
            elif aggregation == 'yearly':
                grouped = df_prepared.groupBy("year")
            else:
                grouped = df_prepared.groupBy("year", "month")

            # Compute aggregations with limit for Colab
            agg_df = grouped.agg(
                spark_sum(value_col).alias("total"),
                avg(value_col).alias("average"),
                count(value_col).alias("count")
            ).orderBy("year", "month" if aggregation != 'yearly' else col("year")) \
             .limit(100)  # Limit results for Colab

            # Collect results
            results_data = agg_df.collect()
            aggregated_data = []

            for row in results_data:
                data_point = {
                    'year': row['year'],
                    'total': float(row['total']) if row['total'] else 0,
                    'average': float(row['average']) if row['average'] else 0,
                    'count': row['count']
                }
                if aggregation in ['daily', 'monthly']:
                    data_point['month'] = row['month']
                if aggregation == 'daily':
                    data_point['day'] = row['day']

                aggregated_data.append(data_point)

            execution_time = time.time() - start_time

            results = {
                'model_type': f'Time Series Aggregation ({aggregation})',
                'date_column': date_col,
                'value_column': value_col,
                'aggregation_period': aggregation,
                'num_periods': len(aggregated_data),
                'aggregated_data': aggregated_data,
                'execution_time_seconds': round(execution_time, 3)
            }

            # Cleanup
            df_prepared.unpersist()

            return results

        except Exception as e:
            return {
                'model_type': f'Time Series Aggregation ({aggregation})',
                'error': str(e),
                'execution_time_seconds': round(time.time() - start_time, 3)
            }
