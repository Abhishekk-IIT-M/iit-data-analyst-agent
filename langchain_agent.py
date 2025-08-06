import os
import pandas as pd
import logging
from typing import Dict, List, Any, Optional
import json

# LangChain imports
try:
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.agents import create_pandas_dataframe_agent
    from langchain.agents.agent_types import AgentType
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)

class LangChainAgent:
    """
    LangChain integration for intelligent data analysis
    """
    
    def __init__(self):
        self.llm = None
        self.chat_model = None
        self._setup_models()
    
    def _setup_models(self):
        """Initialize LangChain models"""
        try:
            if not LANGCHAIN_AVAILABLE:
                logger.warning("LangChain not available. AI insights will be limited.")
                return
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("OpenAI API key not found. AI insights will be limited.")
                return
            
            # Initialize models
            self.llm = OpenAI(
                temperature=0.1,
                openai_api_key=api_key,
                max_tokens=1000
            )
            
            self.chat_model = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.1,
                openai_api_key=api_key,
                max_tokens=1000
            )
            
            logger.info("LangChain models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LangChain models: {str(e)}")
    
    def generate_insights(
        self, 
        df: pd.DataFrame, 
        statistical_results: Dict[str, Any],
        query: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Generate AI-powered insights from data analysis results
        """
        try:
            insights = []
            
            if not self.chat_model:
                # Fallback to rule-based insights
                return self._generate_rule_based_insights(df, statistical_results)
            
            # Create data summary for AI analysis
            data_summary = self._create_data_summary(df, statistical_results)
            
            # Generate general insights
            general_insights = self._generate_general_insights(data_summary)
            if general_insights:
                insights.extend(general_insights)
            
            # Generate specific insights if query provided
            if query:
                specific_insights = self._generate_query_specific_insights(data_summary, query)
                if specific_insights:
                    insights.extend(specific_insights)
            
            # Generate statistical insights
            statistical_insights = self._generate_statistical_insights(statistical_results)
            if statistical_insights:
                insights.extend(statistical_insights)
            
            return insights
            
        except Exception as e:
            logger.error(f"Insight generation failed: {str(e)}")
            return self._generate_rule_based_insights(df, statistical_results)
    
    def extract_structured_data(self, text_content: str) -> Optional[List[Dict[str, Any]]]:
        """
        Extract structured data from unstructured text using AI
        """
        try:
            if not self.chat_model:
                return None
            
            prompt = f"""
            Extract structured data from the following text. 
            Return the data as a JSON array of objects where each object represents a data point.
            If no structured data can be extracted, return an empty array.
            
            Text content:
            {text_content[:2000]}  # Limit text length
            
            Return only valid JSON:
            """
            
            messages = [
                SystemMessage(content="You are a data extraction expert. Extract structured data from text and return as JSON."),
                HumanMessage(content=prompt)
            ]
            
            response = self.chat_model(messages)
            
            # Try to parse JSON response
            try:
                structured_data = json.loads(response.content)
                if isinstance(structured_data, list):
                    return structured_data
            except json.JSONDecodeError:
                logger.warning("Could not parse structured data response as JSON")
            
            return None
            
        except Exception as e:
            logger.error(f"Structured data extraction failed: {str(e)}")
            return None
    
    def create_pandas_agent(self, df: pd.DataFrame) -> Any:
        """
        Create a pandas DataFrame agent for interactive querying
        """
        try:
            if not LANGCHAIN_AVAILABLE or not self.llm:
                return None
            
            agent = create_pandas_dataframe_agent(
                self.llm,
                df,
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
            )
            
            return agent
            
        except Exception as e:
            logger.error(f"Pandas agent creation failed: {str(e)}")
            return None
    
    def _create_data_summary(self, df: pd.DataFrame, statistical_results: Dict[str, Any]) -> str:
        """Create a comprehensive summary of the data for AI analysis"""
        summary_parts = []
        
        # Basic data info
        summary_parts.append(f"Dataset contains {len(df)} rows and {len(df.columns)} columns.")
        summary_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
        
        # Data types
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if numeric_cols:
            summary_parts.append(f"Numeric columns: {', '.join(numeric_cols)}")
        if categorical_cols:
            summary_parts.append(f"Categorical columns: {', '.join(categorical_cols)}")
        
        # Statistical summary
        if 'descriptive_stats' in statistical_results:
            summary_parts.append("Statistical analysis shows:")
            desc_stats = statistical_results['descriptive_stats']
            if isinstance(desc_stats, dict) and 'basic_stats' in desc_stats:
                for col, stats in desc_stats['basic_stats'].items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        summary_parts.append(f"  {col}: mean={stats['mean']:.2f}, std={stats.get('std', 0):.2f}")
        
        # Correlation insights
        if 'correlation_analysis' in statistical_results:
            corr_data = statistical_results['correlation_analysis']
            if 'strong_correlations' in corr_data:
                strong_corrs = corr_data['strong_correlations']
                if strong_corrs:
                    summary_parts.append("Strong correlations found:")
                    for corr in strong_corrs[:3]:  # Top 3
                        summary_parts.append(f"  {corr['variable1']} vs {corr['variable2']}: r={corr['pearson_r']:.2f}")
        
        return '\n'.join(summary_parts)
    
    def _generate_general_insights(self, data_summary: str) -> List[Dict[str, str]]:
        """Generate general insights about the dataset"""
        try:
            prompt = f"""
            Based on the following data summary, provide 3-5 key insights about the dataset.
            Focus on data quality, patterns, and potential areas of interest for analysis.
            
            Data Summary:
            {data_summary}
            
            Provide insights in this format:
            1. [Insight about data structure/quality]
            2. [Insight about patterns or trends]
            3. [Insight about potential analysis opportunities]
            """
            
            messages = [
                SystemMessage(content="You are a data analyst providing insights about datasets."),
                HumanMessage(content=prompt)
            ]
            
            response = self.chat_model(messages)
            insights = self._parse_insights_response(response.content, "general")
            
            return insights
            
        except Exception as e:
            logger.error(f"General insight generation failed: {str(e)}")
            return []
    
    def _generate_query_specific_insights(self, data_summary: str, query: str) -> List[Dict[str, str]]:
        """Generate insights specific to user query"""
        try:
            prompt = f"""
            Based on the data summary and user query, provide specific insights that address the query.
            
            Data Summary:
            {data_summary}
            
            User Query: {query}
            
            Provide 2-3 specific insights that directly relate to the user's question.
            """
            
            messages = [
                SystemMessage(content="You are a data analyst answering specific questions about datasets."),
                HumanMessage(content=prompt)
            ]
            
            response = self.chat_model(messages)
            insights = self._parse_insights_response(response.content, "query_specific")
            
            return insights
            
        except Exception as e:
            logger.error(f"Query-specific insight generation failed: {str(e)}")
            return []
    
    def _generate_statistical_insights(self, statistical_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate insights from statistical analysis results"""
        insights = []
        
        try:
            # Correlation insights
            if 'correlation_analysis' in statistical_results:
                corr_data = statistical_results['correlation_analysis']
                if 'strong_correlations' in corr_data and corr_data['strong_correlations']:
                    insights.append({
                        'type': 'correlation',
                        'title': 'Strong Correlations Detected',
                        'content': f"Found {len(corr_data['strong_correlations'])} strong correlations in the data, "
                                 f"suggesting potential relationships between variables that warrant further investigation."
                    })
            
            # Outlier insights
            if 'outlier_analysis' in statistical_results:
                outlier_data = statistical_results['outlier_analysis']
                high_outlier_cols = [col for col, data in outlier_data.items() 
                                   if isinstance(data, dict) and 
                                   data.get('iqr_method', {}).get('percentage', 0) > 5]
                
                if high_outlier_cols:
                    insights.append({
                        'type': 'outliers',
                        'title': 'Significant Outliers Detected',
                        'content': f"Variables {', '.join(high_outlier_cols)} contain significant outliers (>5% of data). "
                                 f"Consider investigating these anomalies or applying outlier treatment."
                    })
            
            # Normality insights
            if 'normality_tests' in statistical_results:
                normality_data = statistical_results['normality_tests']
                non_normal_vars = [var for var, tests in normality_data.items()
                                 if isinstance(tests, dict) and 
                                 not tests.get('shapiro_wilk', {}).get('is_normal', True)]
                
                if non_normal_vars:
                    insights.append({
                        'type': 'normality',
                        'title': 'Non-normal Distributions Detected',
                        'content': f"Variables {', '.join(non_normal_vars)} do not follow normal distributions. "
                                 f"Consider data transformation or non-parametric tests for these variables."
                    })
            
            return insights
            
        except Exception as e:
            logger.error(f"Statistical insight generation failed: {str(e)}")
            return []
    
    def _parse_insights_response(self, response: str, insight_type: str) -> List[Dict[str, str]]:
        """Parse AI response into structured insights"""
        insights = []
        
        try:
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith(('1.', '2.', '3.', '4.', '5.')) or line.startswith('-')):
                    content = line.split('.', 1)[-1].strip() if '.' in line else line.strip('- ')
                    if content:
                        insights.append({
                            'type': insight_type,
                            'title': f"{insight_type.title()} Insight",
                            'content': content
                        })
            
            return insights
            
        except Exception as e:
            logger.error(f"Insight parsing failed: {str(e)}")
            return []
    
    def _generate_rule_based_insights(self, df: pd.DataFrame, statistical_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Fallback rule-based insights when AI is not available"""
        insights = []
        
        # Basic data insights
        insights.append({
            'type': 'basic',
            'title': 'Dataset Overview',
            'content': f"Dataset contains {len(df)} rows and {len(df.columns)} columns. "
                      f"Data types include {len(df.select_dtypes(include=['number']).columns)} numeric "
                      f"and {len(df.select_dtypes(include=['object']).columns)} categorical variables."
        })
        
        # Missing data insights
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            high_missing = missing_data[missing_data > len(df) * 0.1]
            if not high_missing.empty:
                insights.append({
                    'type': 'data_quality',
                    'title': 'Missing Data Alert',
                    'content': f"High missing data detected in columns: {', '.join(high_missing.index)}. "
                              f"Consider data imputation or removal of these variables."
                })
        
        return insights
