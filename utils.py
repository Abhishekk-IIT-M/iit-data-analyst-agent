import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Union
import os
from werkzeug.datastructures import FileStorage
import json

logger = logging.getLogger(__name__)

def validate_file(file: FileStorage, allowed_extensions: set) -> Dict[str, Union[bool, str]]:
    """
    Validate uploaded file
    """
    try:
        if file.filename == '':
            return {'valid': False, 'error': 'No file selected'}
        
        # Check file extension
        if not '.' in file.filename:
            return {'valid': False, 'error': 'File has no extension'}
        
        extension = file.filename.rsplit('.', 1)[1].lower()
        if extension not in allowed_extensions:
            return {'valid': False, 'error': f'File extension .{extension} not allowed. Allowed: {", ".join(allowed_extensions)}'}
        
        # Check file size (basic check, Flask config handles the rest)
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)  # Reset file pointer
        
        if size == 0:
            return {'valid': False, 'error': 'File is empty'}
        
        # Additional validation for CSV files
        if extension == 'csv':
            # Try to read first few lines to validate CSV format
            try:
                file.seek(0)
                sample = file.read(1024).decode('utf-8')
                file.seek(0)  # Reset file pointer
                
                # Basic CSV validation - check for common delimiters
                if ',' not in sample and ';' not in sample and '\t' not in sample:
                    return {'valid': False, 'error': 'File does not appear to be a valid CSV format'}
                    
            except UnicodeDecodeError:
                return {'valid': False, 'error': 'File encoding is not supported'}
        
        return {'valid': True, 'error': None}
        
    except Exception as e:
        logger.error(f"File validation failed: {str(e)}")
        return {'valid': False, 'error': f'File validation error: {str(e)}'}

def detect_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Detect and categorize data types in DataFrame
    """
    try:
        data_types = {}
        
        for column in df.columns:
            series = df[column]
            
            if pd.api.types.is_numeric_dtype(series):
                if pd.api.types.is_integer_dtype(series):
                    data_types[column] = 'integer'
                else:
                    data_types[column] = 'float'
            elif pd.api.types.is_datetime64_any_dtype(series):
                data_types[column] = 'datetime'
            elif pd.api.types.is_bool_dtype(series):
                data_types[column] = 'boolean'
            else:
                # Check if string column might be categorical
                unique_ratio = len(series.unique()) / len(series)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    data_types[column] = 'categorical'
                else:
                    data_types[column] = 'text'
        
        return data_types
        
    except Exception as e:
        logger.error(f"Data type detection failed: {str(e)}")
        return {}

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess DataFrame
    """
    try:
        df_cleaned = df.copy()
        
        # Remove completely empty rows and columns
        df_cleaned = df_cleaned.dropna(how='all')  # Remove rows with all NaN
        df_cleaned = df_cleaned.dropna(axis=1, how='all')  # Remove columns with all NaN
        
        # Clean column names
        df_cleaned.columns = df_cleaned.columns.str.strip()  # Remove whitespace
        df_cleaned.columns = df_cleaned.columns.str.replace(r'[^\w\s]', '', regex=True)  # Remove special chars
        df_cleaned.columns = df_cleaned.columns.str.replace(r'\s+', '_', regex=True)  # Replace spaces with underscores
        
        # Handle numeric columns
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # Replace infinite values with NaN
            df_cleaned[col] = df_cleaned[col].replace([np.inf, -np.inf], np.nan)
        
        # Handle text columns
        text_columns = df_cleaned.select_dtypes(include=['object']).columns
        for col in text_columns:
            # Strip whitespace and handle empty strings
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
            df_cleaned[col] = df_cleaned[col].replace(['', 'nan', 'None', 'NULL'], np.nan)
        
        logger.info(f"Data cleaning completed. Shape: {df_cleaned.shape}")
        return df_cleaned
        
    except Exception as e:
        logger.error(f"Data cleaning failed: {str(e)}")
        return df

def create_error_response(message: str, details: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create standardized error response
    """
    response = {
        'success': False,
        'error': True,
        'message': message,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    if details:
        response['details'] = details
    
    return response

def create_success_response(data: Any, message: str = "Analysis completed successfully") -> Dict[str, Any]:
    """
    Create standardized success response
    """
    return {
        'success': True,
        'error': False,
        'message': message,
        'data': convert_numpy_types(data),
        'timestamp': pd.Timestamp.now().isoformat()
    }

def convert_numpy_types(obj: Any) -> Any:
    """
    Convert NumPy and Pandas types to JSON serializable types
    """
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

def summarize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a comprehensive summary of a DataFrame
    """
    try:
        summary = {
            'shape': {
                'rows': len(df),
                'columns': len(df.columns)
            },
            'columns': {
                'names': list(df.columns),
                'types': df.dtypes.astype(str).to_dict()
            },
            'missing_data': {
                'total_missing': int(df.isnull().sum().sum()),
                'by_column': df.isnull().sum().to_dict(),
                'percentage_missing': (df.isnull().sum() / len(df) * 100).to_dict()
            },
            'memory_usage': {
                'total_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
                'by_column_mb': (df.memory_usage(deep=True) / 1024 / 1024).round(2).to_dict()
            }
        }
        
        # Numeric summary
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            summary['numeric_summary'] = {
                'count': len(numeric_df.columns),
                'basic_stats': numeric_df.describe().to_dict()
            }
        
        # Categorical summary
        categorical_df = df.select_dtypes(include=['object', 'category'])
        if not categorical_df.empty:
            summary['categorical_summary'] = {
                'count': len(categorical_df.columns),
                'unique_values': {col: int(categorical_df[col].nunique()) for col in categorical_df.columns},
                'most_frequent': {col: categorical_df[col].mode().iloc[0] if len(categorical_df[col].mode()) > 0 else None 
                                for col in categorical_df.columns}
            }
        
        return summary
        
    except Exception as e:
        logger.error(f"DataFrame summarization failed: {str(e)}")
        return {'error': str(e)}

def format_number(value: Union[int, float], decimals: int = 2) -> str:
    """
    Format numbers for display
    """
    try:
        if pd.isna(value) or value is None:
            return "N/A"
        
        if isinstance(value, (int, np.integer)):
            return f"{value:,}"
        elif isinstance(value, (float, np.floating)):
            if abs(value) >= 1000000:
                return f"{value/1000000:.{decimals}f}M"
            elif abs(value) >= 1000:
                return f"{value/1000:.{decimals}f}K"
            else:
                return f"{value:.{decimals}f}"
        else:
            return str(value)
            
    except Exception:
        return str(value)

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, handling division by zero
    """
    try:
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default
        return numerator / denominator
    except Exception:
        return default

def validate_s3_path(s3_path: str) -> bool:
    """
    Validate S3 path format
    """
    try:
        if not s3_path:
            return False
        
        # Remove s3:// prefix if present
        path = s3_path.replace('s3://', '')
        
        # Check for bucket/key format
        parts = path.split('/', 1)
        if len(parts) < 2:
            return False
        
        bucket, key = parts
        
        # Basic bucket name validation
        if not bucket or len(bucket) < 3 or len(bucket) > 63:
            return False
        
        # Basic key validation
        if not key:
            return False
        
        return True
        
    except Exception:
        return False
