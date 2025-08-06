import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Union
import traceback

from web_scraper import get_website_text_content
from statistical_analyzer import StatisticalAnalyzer
from visualization_generator import VisualizationGenerator
from duckdb_manager import DuckDBManager
from langchain_agent import LangChainAgent
from utils import detect_data_types, clean_data

logger = logging.getLogger(__name__)

class DataAnalystAgent:
    """
    Main orchestrator for data analysis operations
    Coordinates web scraping, statistical analysis, and visualization
    """
    
    def __init__(self):
        self.statistical_analyzer = StatisticalAnalyzer()
        self.visualization_generator = VisualizationGenerator()
        self.duckdb_manager = DuckDBManager()
        self.langchain_agent = LangChainAgent()
        
    def analyze(
        self,
        file_path: Optional[str] = None,
        web_url: Optional[str] = None,
        s3_path: Optional[str] = None,
        analysis_type: str = 'basic',
        query: Optional[str] = None,
        visualization_type: str = 'auto',
        statistical_tests: List[str] = None
    ) -> Dict[str, Any]:
        """
        Main analysis method that orchestrates the entire process
        """
        logger.info(f"Starting analysis - type: {analysis_type}")
        
        try:
            results = {
                'analysis_type': analysis_type,
                'data_summary': {},
                'statistical_analysis': {},
                'visualizations': [],
                'insights': [],
                'metadata': {}
            }
            
            # Step 1: Load and prepare data
            df = self._load_data(file_path, web_url, s3_path, query)
            if df is None or df.empty:
                raise ValueError("No data could be loaded from the provided sources")
            
            logger.info(f"Loaded data with shape: {df.shape}")
            
            # Step 2: Clean and analyze data structure
            df_cleaned = clean_data(df)
            data_types = detect_data_types(df_cleaned)
            
            results['data_summary'] = {
                'rows': len(df_cleaned),
                'columns': len(df_cleaned.columns),
                'column_names': list(df_cleaned.columns),
                'data_types': data_types,
                'missing_values': df_cleaned.isnull().sum().to_dict(),
                'memory_usage': df_cleaned.memory_usage(deep=True).sum()
            }
            
            # Step 3: Statistical Analysis
            if analysis_type in ['basic', 'comprehensive', 'statistical']:
                stats_results = self.statistical_analyzer.analyze(
                    df_cleaned, 
                    analysis_type=analysis_type,
                    tests=statistical_tests or []
                )
                results['statistical_analysis'] = stats_results
            
            # Step 4: Generate Visualizations
            if visualization_type != 'none':
                viz_results = self.visualization_generator.generate_visualizations(
                    df_cleaned,
                    viz_type=visualization_type,
                    data_types=data_types
                )
                results['visualizations'] = viz_results
            
            # Step 5: LangChain AI Insights
            if query or analysis_type == 'comprehensive':
                ai_insights = self.langchain_agent.generate_insights(
                    df_cleaned,
                    results['statistical_analysis'],
                    query=query
                )
                results['insights'] = ai_insights
            
            # Step 6: Metadata
            results['metadata'] = {
                'processing_time': None,  # Could add timing
                'data_sources': {
                    'file': file_path is not None,
                    'web': web_url is not None,
                    's3': s3_path is not None
                },
                'analysis_parameters': {
                    'analysis_type': analysis_type,
                    'visualization_type': visualization_type,
                    'statistical_tests': statistical_tests
                }
            }
            
            logger.info("Analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _load_data(
        self, 
        file_path: Optional[str], 
        web_url: Optional[str], 
        s3_path: Optional[str],
        query: Optional[str]
    ) -> Optional[pd.DataFrame]:
        """Load data from various sources"""
        
        try:
            # Priority: file > s3 > web
            if file_path:
                return self._load_from_file(file_path)
            elif s3_path:
                return self._load_from_s3(s3_path, query)
            elif web_url:
                return self._load_from_web(web_url)
            else:
                raise ValueError("No valid data source provided")
                
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise
    
    def _load_from_file(self, file_path: str) -> pd.DataFrame:
        """Load data from uploaded file"""
        logger.info(f"Loading data from file: {file_path}")
        
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def _load_from_s3(self, s3_path: str, query: Optional[str] = None) -> pd.DataFrame:
        """Load data from S3 using DuckDB"""
        logger.info(f"Loading data from S3: {s3_path}")
        
        if query:
            return self.duckdb_manager.query_s3(s3_path, query)
        else:
            # Default query to select all
            return self.duckdb_manager.query_s3(s3_path, f"SELECT * FROM '{s3_path}'")
    
    def _load_from_web(self, web_url: str) -> pd.DataFrame:
        """Load and parse data from web URL"""
        logger.info(f"Loading data from web: {web_url}")
        
        # Get text content from web page
        text_content = get_website_text_content(web_url)
        
        if not text_content:
            raise ValueError(f"Could not extract content from URL: {web_url}")
        
        # Try to parse as structured data using LangChain
        structured_data = self.langchain_agent.extract_structured_data(text_content)
        
        if structured_data:
            return pd.DataFrame(structured_data)
        else:
            # If no structured data, return text analysis
            return pd.DataFrame({
                'content': [text_content],
                'url': [web_url],
                'word_count': [len(text_content.split())],
                'char_count': [len(text_content)]
            })
