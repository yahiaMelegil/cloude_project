import os
import pandas as pd
import json
from pathlib import Path
from typing import Tuple, Optional
from utils.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE

class DataValidator:
    """Validates uploaded files and data"""
    
    @staticmethod
    def validate_file_extension(filename: str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    @staticmethod
    def validate_file_size(file) -> bool:
        """Check if file size is within limits"""
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        return file_size <= MAX_FILE_SIZE
    
    @staticmethod
    def load_data(file_path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Load data from various file formats
        Returns: (dataframe, error_message)
        """
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension == '.json':
                df = pd.read_json(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_extension == '.txt':
                # Try to read as CSV with tab delimiter
                try:
                    df = pd.read_csv(file_path, sep='\t')
                except:
                    df = pd.read_csv(file_path)
            else:
                return None, f"Unsupported file format: {file_extension}"
            
            return df, None
            
        except Exception as e:
            return None, f"Error loading file: {str(e)}"
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """Validate loaded dataframe"""
        if df is None or df.empty:
            return False, "Dataset is empty"
        
        if len(df) < 10:
            return False, "Dataset too small (minimum 10 rows required)"
        
        if len(df.columns) < 2:
            return False, "Dataset must have at least 2 columns"
        
        return True, None
    
    @staticmethod
    def get_data_info(df: pd.DataFrame) -> dict:
        """Get basic information about the dataset"""
        return {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
            'has_nulls': df.isnull().any().any()
        }