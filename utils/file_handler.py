import os
import shutil
from pathlib import Path
from datetime import datetime
from utils.config import UPLOAD_DIR, RESULTS_DIR

class FileHandler:
    """Handles file operations for uploads and results"""
    
    @staticmethod
    def save_uploaded_file(uploaded_file) -> str:
        """
        Save uploaded file to upload directory
        Returns: path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{uploaded_file.name}"
        file_path = UPLOAD_DIR / filename
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return str(file_path)
    
    @staticmethod
    def save_results(results: dict, base_filename: str) -> str:
        """
        Save processing results to results directory
        Returns: path to saved results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"{timestamp}_{base_filename}_results.json"
        result_path = RESULTS_DIR / result_filename
        
        import json
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        return str(result_path)
    
    @staticmethod
    def get_uploaded_files() -> list:
        """Get list of all uploaded files"""
        if not UPLOAD_DIR.exists():
            return []
        
        files = []
        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file():
                files.append({
                    'name': file_path.name,
                    'path': str(file_path),
                    'size': file_path.stat().st_size,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
                })
        
        return sorted(files, key=lambda x: x['modified'], reverse=True)
    
    @staticmethod
    def get_result_files() -> list:
        """Get list of all result files"""
        if not RESULTS_DIR.exists():
            return []
        
        files = []
        for file_path in RESULTS_DIR.iterdir():
            if file_path.is_file() and file_path.suffix == '.json':
                files.append({
                    'name': file_path.name,
                    'path': str(file_path),
                    'size': file_path.stat().st_size,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
                })
        
        return sorted(files, key=lambda x: x['modified'], reverse=True)
    
    @staticmethod
    def cleanup_old_files(days: int = 7):
        """Remove files older than specified days"""
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        for directory in [UPLOAD_DIR, RESULTS_DIR]:
            if directory.exists():
                for file_path in directory.iterdir():
                    if file_path.is_file() and file_path.stat().st_mtime < cutoff:
                        file_path.unlink()
    
    @staticmethod
    def get_file_size_str(size_bytes: int) -> str:
        """Convert bytes to human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"