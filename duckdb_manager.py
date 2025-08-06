import duckdb
import pandas as pd
import os
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class DuckDBManager:
    """
    Manage DuckDB connections and queries for S3 data analysis
    """
    
    def __init__(self):
        self.conn = None
        self._setup_connection()
    
    def _setup_connection(self):
        """Initialize DuckDB connection with S3 configuration"""
        try:
            self.conn = duckdb.connect()
            
            # Install and load httpfs extension for S3 access
            self.conn.execute("INSTALL httpfs;")
            self.conn.execute("LOAD httpfs;")
            
            # Configure S3 credentials if available
            aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
            
            if aws_access_key and aws_secret_key:
                self.conn.execute(f"SET s3_region='{aws_region}';")
                self.conn.execute(f"SET s3_access_key_id='{aws_access_key}';")
                self.conn.execute(f"SET s3_secret_access_key='{aws_secret_key}';")
                logger.info("S3 credentials configured")
            else:
                logger.warning("S3 credentials not found in environment variables")
            
        except Exception as e:
            logger.error(f"Failed to setup DuckDB connection: {str(e)}")
            raise
    
    def query_s3(self, s3_path: str, query: str = None) -> pd.DataFrame:
        """
        Query Parquet files from S3
        """
        try:
            if not query:
                query = f"SELECT * FROM '{s3_path}'"
            
            # Handle different S3 path formats
            if not s3_path.startswith('s3://'):
                s3_path = f"s3://{s3_path}"
            
            # Replace placeholder in query if needed
            query = query.replace('{s3_path}', s3_path)
            
            logger.info(f"Executing query: {query}")
            result = self.conn.execute(query).df()
            
            logger.info(f"Query returned {len(result)} rows")
            return result
            
        except Exception as e:
            logger.error(f"S3 query failed: {str(e)}")
            raise
    
    def query_local_parquet(self, file_path: str, query: str = None) -> pd.DataFrame:
        """
        Query local Parquet files
        """
        try:
            if not query:
                query = f"SELECT * FROM '{file_path}'"
            
            logger.info(f"Executing local query: {query}")
            result = self.conn.execute(query).df()
            
            logger.info(f"Query returned {len(result)} rows")
            return result
            
        except Exception as e:
            logger.error(f"Local parquet query failed: {str(e)}")
            raise
    
    def analyze_parquet_schema(self, s3_path: str) -> Dict[str, Any]:
        """
        Analyze the schema of a Parquet file
        """
        try:
            if not s3_path.startswith('s3://'):
                s3_path = f"s3://{s3_path}"
            
            # Get schema information
            schema_query = f"DESCRIBE SELECT * FROM '{s3_path}' LIMIT 0"
            schema_result = self.conn.execute(schema_query).df()
            
            # Get basic statistics
            stats_query = f"""
            SELECT 
                COUNT(*) as row_count,
                COUNT(DISTINCT *) as unique_rows
            FROM '{s3_path}'
            """
            
            stats_result = self.conn.execute(stats_query).df()
            
            return {
                'schema': schema_result.to_dict('records'),
                'statistics': stats_result.to_dict('records')[0] if len(stats_result) > 0 else {}
            }
            
        except Exception as e:
            logger.error(f"Schema analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def execute_aggregation_query(self, s3_path: str, group_by_cols: List[str], 
                                 agg_cols: List[str], agg_functions: List[str] = None) -> pd.DataFrame:
        """
        Execute aggregation queries on S3 data
        """
        try:
            if not s3_path.startswith('s3://'):
                s3_path = f"s3://{s3_path}"
            
            if not agg_functions:
                agg_functions = ['COUNT', 'AVG', 'SUM', 'MIN', 'MAX']
            
            # Build aggregation query
            agg_parts = []
            for col in agg_cols:
                for func in agg_functions:
                    agg_parts.append(f"{func}({col}) as {func.lower()}_{col}")
            
            group_by_str = ', '.join(group_by_cols)
            agg_str = ', '.join(agg_parts)
            
            query = f"""
            SELECT 
                {group_by_str},
                {agg_str}
            FROM '{s3_path}'
            GROUP BY {group_by_str}
            ORDER BY {group_by_cols[0]}
            """
            
            logger.info(f"Executing aggregation query: {query}")
            result = self.conn.execute(query).df()
            
            return result
            
        except Exception as e:
            logger.error(f"Aggregation query failed: {str(e)}")
            raise
    
    def get_sample_data(self, s3_path: str, sample_size: int = 1000) -> pd.DataFrame:
        """
        Get a sample of data from S3 for analysis
        """
        try:
            if not s3_path.startswith('s3://'):
                s3_path = f"s3://{s3_path}"
            
            query = f"SELECT * FROM '{s3_path}' LIMIT {sample_size}"
            
            logger.info(f"Getting sample data: {query}")
            result = self.conn.execute(query).df()
            
            return result
            
        except Exception as e:
            logger.error(f"Sample data query failed: {str(e)}")
            raise
    
    def close_connection(self):
        """Close DuckDB connection"""
        try:
            if self.conn:
                self.conn.close()
                logger.info("DuckDB connection closed")
        except Exception as e:
            logger.error(f"Error closing DuckDB connection: {str(e)}")
    
    def __del__(self):
        """Destructor to ensure connection is closed"""
        self.close_connection()
