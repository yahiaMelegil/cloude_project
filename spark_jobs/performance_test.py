# performance_test.py - UPDATED FOR GOOGLE COLAB
from pyspark.sql import SparkSession
import time
from typing import Dict, List
import traceback

class PerformanceTester:
    """Test ML models with different cluster configurations - Optimized for Colab"""
    
    def __init__(self):
        self.results = []
    
    def create_spark_session(self, config: Dict) -> SparkSession:
        """
        Create Spark session with specific configuration for Colab
        """
        # First stop any existing sessions
        try:
            SparkSession.builder.getOrCreate().stop()
        except:
            pass
        
        import time
        time.sleep(1)
        
        spark = SparkSession.builder \
            .appName(f"PerformanceTest_{config['name']}") \
            .master(f"local[{config['cores']}]") \
            .config("spark.executor.memory", config['memory']) \
            .config("spark.driver.memory", config['memory']) \
            .config("spark.ui.showConsoleProgress", "false") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.driver.maxResultSize", "1g") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("ERROR")
        return spark
    
    def run_performance_test(self, data_path: str, ml_function, 
                            ml_params: Dict) -> List[Dict]:
        """
        Run the same ML job with different configurations
        and measure performance - Optimized for Colab
        """
        performance_results = []
        
        # Test configurations optimized for Colab
        test_configs = [
            {'name': '1_core', 'executors': 1, 'cores': 1, 'memory': '1g'},
            {'name': '2_cores', 'executors': 1, 'cores': 2, 'memory': '1g'},
            {'name': '4_cores', 'executors': 1, 'cores': 4, 'memory': '1g'}
        ]
        
        for config in test_configs:
            print(f"\n{'='*60}")
            print(f"Testing with configuration: {config['name']}")
            print(f"Cores: {config['cores']}, Memory: {config['memory']}")
            print(f"{'='*60}\n")
            
            # Create Spark session with specific config
            spark = self.create_spark_session(config)
            
            try:
                # Load data
                start_load = time.time()
                df = spark.read.csv(data_path, header=True, inferSchema=True)
                load_time = time.time() - start_load
                
                # Run ML model
                from spark_jobs.ml_models import MLModels
                ml_models = MLModels(spark)
                
                start_ml = time.time()
                result = ml_function(ml_models, df, **ml_params)
                ml_time = time.time() - start_ml
                
                # Store results
                performance_results.append({
                    'configuration': config['name'],
                    'num_cores': config['cores'],
                    'memory': config['memory'],
                    'data_load_time': round(load_time, 3),
                    'ml_execution_time': round(ml_time, 3),
                    'total_time': round(load_time + ml_time, 3),
                    'model_result': result
                })
                
                print(f"✓ Completed {config['name']} in {load_time + ml_time:.2f}s")
                
            except Exception as e:
                print(f"✗ Error with config {config['name']}: {str(e)}")
                performance_results.append({
                    'configuration': config['name'],
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
            
            finally:
                # Always stop the Spark session
                try:
                    spark.stop()
                except:
                    pass
                
                time.sleep(2)  # Wait before starting next session
        
        return performance_results
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """
        Calculate speedup and efficiency metrics
        """
        if not results or 'error' in results[0]:
            return {'error': 'No valid results to calculate metrics'}
        
        # Find baseline (first valid result)
        baseline = next((r for r in results if 'error' not in r), None)
        if not baseline:
            return {'error': 'No valid results found'}
        
        baseline_time = baseline['total_time']
        
        metrics = []
        for result in results:
            if 'error' not in result:
                speedup = baseline_time / result['total_time']
                # Efficiency = Speedup / Number of cores
                efficiency = speedup / result['num_cores'] if result['num_cores'] > 0 else 0
                
                metrics.append({
                    'configuration': result['configuration'],
                    'num_cores': result['num_cores'],
                    'execution_time': result['total_time'],
                    'speedup': round(speedup, 3),
                    'efficiency': round(efficiency * 100, 2)  # As percentage
                })
        
        return {
            'baseline_time': baseline_time,
            'performance_metrics': metrics
        }
    
    def generate_performance_report(self, results: List[Dict]) -> Dict:
        """
        Generate comprehensive performance report
        """
        metrics = self.calculate_metrics(results)
        
        report = {
            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'configurations_tested': len(results),
            'valid_results': len([r for r in results if 'error' not in r]),
            'raw_results': results,
            'performance_metrics': metrics,
            'summary': self._generate_summary(metrics)
        }
        
        return report
    
    def _generate_summary(self, metrics: Dict) -> Dict:
        """
        Generate summary of performance results
        """
        if 'error' in metrics:
            return {'error': metrics['error']}
        
        perf_metrics = metrics['performance_metrics']
        
        if not perf_metrics:
            return {'error': 'No performance metrics available'}
        
        # Find best configuration
        best_speedup = max(perf_metrics, key=lambda x: x['speedup'])
        best_efficiency = max(perf_metrics, key=lambda x: x['efficiency'])
        
        # Calculate average speedup improvement
        speedups = [m['speedup'] for m in perf_metrics]
        avg_speedup = sum(speedups) / len(speedups)
        
        summary = {
            'baseline_time_seconds': metrics['baseline_time'],
            'best_speedup_config': best_speedup['configuration'],
            'best_speedup_value': best_speedup['speedup'],
            'best_efficiency_config': best_efficiency['configuration'],
            'best_efficiency_value': best_efficiency['efficiency'],
            'average_speedup': round(avg_speedup, 3),
            'scalability_assessment': self._assess_scalability(perf_metrics)
        }
        
        return summary
    
    def _assess_scalability(self, metrics: List[Dict]) -> str:
        """
        Assess how well the system scales
        """
        if len(metrics) < 2:
            return "Insufficient data for scalability assessment"
        
        # Sort by number of cores
        metrics_sorted = sorted(metrics, key=lambda x: x['num_cores'])
        
        # Check if speedup increases with more cores
        speedups = [m['speedup'] for m in metrics_sorted]
        
        if all(speedups[i] <= speedups[i+1] for i in range(len(speedups)-1)):
            if speedups[-1] / speedups[0] > 2.5:
                return "Excellent scalability - near linear speedup"
            elif speedups[-1] / speedups[0] > 1.5:
                return "Good scalability - significant performance improvement"
            else:
                return "Moderate scalability - some performance improvement"
        else:
            return "Poor scalability - diminishing returns with more resources"
    
    def print_performance_table(self, metrics: Dict):
        """
        Print performance results in table format
        """
        if 'error' in metrics:
            print(f"Error: {metrics['error']}")
            return
        
        print("\n" + "="*80)
        print("PERFORMANCE TEST RESULTS")
        print("="*80)
        
        print(f"\n{'Configuration':<20} {'Cores':<10} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<12}")
        print("-"*80)
        
        for m in metrics['performance_metrics']:
            print(f"{m['configuration']:<20} {m['num_cores']:<10} "
                  f"{m['execution_time']:<12.3f} {m['speedup']:<10.3f} "
                  f"{m['efficiency']:<12.2f}%")
        
        print("-"*80)
        print(f"\nBaseline time: {metrics['baseline_time']:.3f} seconds")
        print("="*80)