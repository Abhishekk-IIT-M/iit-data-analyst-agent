import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from io import BytesIO
import base64
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Set matplotlib to use Agg backend for server environments
plt.switch_backend('Agg')
sns.set_style("whitegrid")

class VisualizationGenerator:
    """
    Generate various types of data visualizations
    Returns base64-encoded images
    """
    
    def __init__(self):
        # Set default figure parameters
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.pad_inches'] = 0.2
    
    def generate_visualizations(
        self, 
        df: pd.DataFrame, 
        viz_type: str = 'auto',
        data_types: Dict[str, str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate appropriate visualizations based on data
        """
        try:
            visualizations = []
            
            if viz_type == 'auto':
                # Generate appropriate visualizations based on data characteristics
                visualizations.extend(self._auto_generate_visualizations(df, data_types))
            else:
                # Generate specific visualization type
                viz_result = self._generate_specific_visualization(df, viz_type)
                if viz_result:
                    visualizations.append(viz_result)
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {str(e)}")
            return [{'error': str(e)}]
    
    def _auto_generate_visualizations(self, df: pd.DataFrame, data_types: Dict[str, str]) -> List[Dict[str, Any]]:
        """Automatically generate appropriate visualizations"""
        visualizations = []
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Histogram for numeric data
            if numeric_cols:
                hist_viz = self._create_histogram(df, numeric_cols[:4])  # Limit to first 4 columns
                if hist_viz:
                    visualizations.append(hist_viz)
            
            # Box plot for numeric data
            if len(numeric_cols) > 0:
                box_viz = self._create_box_plot(df, numeric_cols[:3])  # Limit to first 3 columns
                if box_viz:
                    visualizations.append(box_viz)
            
            # Correlation heatmap if multiple numeric columns
            if len(numeric_cols) > 1:
                corr_viz = self._create_correlation_heatmap(df, numeric_cols)
                if corr_viz:
                    visualizations.append(corr_viz)
            
            # Bar chart for categorical data
            if categorical_cols:
                bar_viz = self._create_bar_chart(df, categorical_cols[0])
                if bar_viz:
                    visualizations.append(bar_viz)
            
            # Scatter plot if we have at least 2 numeric columns
            if len(numeric_cols) >= 2:
                scatter_viz = self._create_scatter_plot(df, numeric_cols[0], numeric_cols[1])
                if scatter_viz:
                    visualizations.append(scatter_viz)
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Auto visualization generation failed: {str(e)}")
            return []
    
    def _generate_specific_visualization(self, df: pd.DataFrame, viz_type: str) -> Optional[Dict[str, Any]]:
        """Generate a specific type of visualization"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if viz_type == 'histogram' and numeric_cols:
                return self._create_histogram(df, numeric_cols)
            elif viz_type == 'boxplot' and numeric_cols:
                return self._create_box_plot(df, numeric_cols)
            elif viz_type == 'correlation' and len(numeric_cols) > 1:
                return self._create_correlation_heatmap(df, numeric_cols)
            elif viz_type == 'scatter' and len(numeric_cols) >= 2:
                return self._create_scatter_plot(df, numeric_cols[0], numeric_cols[1])
            elif viz_type == 'bar' and categorical_cols:
                return self._create_bar_chart(df, categorical_cols[0])
            else:
                return None
                
        except Exception as e:
            logger.error(f"Specific visualization generation failed: {str(e)}")
            return None
    
    def _create_histogram(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """Create histogram for numeric columns"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.ravel()
            
            for i, col in enumerate(columns[:4]):
                if i < len(columns):
                    data = df[col].dropna()
                    axes[i].hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
                    axes[i].grid(True, alpha=0.3)
                else:
                    axes[i].set_visible(False)
            
            plt.tight_layout()
            
            return {
                'type': 'histogram',
                'title': 'Distribution Analysis',
                'description': f'Histograms showing the distribution of numeric variables',
                'image': self._fig_to_base64(fig),
                'columns_analyzed': columns[:4]
            }
            
        except Exception as e:
            logger.error(f"Histogram creation failed: {str(e)}")
            return None
        finally:
            plt.close()
    
    def _create_box_plot(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """Create box plot for numeric columns"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            data_to_plot = [df[col].dropna() for col in columns[:3]]  # Limit to 3 columns
            
            ax.boxplot(data_to_plot, labels=columns[:3])
            ax.set_title('Box Plot Analysis')
            ax.set_ylabel('Values')
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return {
                'type': 'boxplot',
                'title': 'Box Plot Analysis',
                'description': 'Box plots showing quartiles, outliers, and distribution shape',
                'image': self._fig_to_base64(fig),
                'columns_analyzed': columns[:3]
            }
            
        except Exception as e:
            logger.error(f"Box plot creation failed: {str(e)}")
            return None
        finally:
            plt.close()
    
    def _create_correlation_heatmap(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """Create correlation heatmap"""
        try:
            # Calculate correlation matrix
            corr_matrix = df[columns].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create heatmap
            im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            
            # Set ticks and labels
            ax.set_xticks(range(len(columns)))
            ax.set_yticks(range(len(columns)))
            ax.set_xticklabels(columns, rotation=45, ha='right')
            ax.set_yticklabels(columns)
            
            # Add correlation values to cells
            for i in range(len(columns)):
                for j in range(len(columns)):
                    ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                           ha='center', va='center',
                           color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
            
            # Add colorbar
            cbar = plt.colorbar(im)
            cbar.set_label('Correlation Coefficient')
            
            ax.set_title('Correlation Matrix Heatmap')
            plt.tight_layout()
            
            return {
                'type': 'correlation_heatmap',
                'title': 'Correlation Analysis',
                'description': 'Heatmap showing correlations between numeric variables',
                'image': self._fig_to_base64(fig),
                'columns_analyzed': columns
            }
            
        except Exception as e:
            logger.error(f"Correlation heatmap creation failed: {str(e)}")
            return None
        finally:
            plt.close()
    
    def _create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str) -> Dict[str, Any]:
        """Create scatter plot"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Remove NaN values
            data = df[[x_col, y_col]].dropna()
            
            ax.scatter(data[x_col], data[y_col], alpha=0.6, s=50)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(data) > 1:
                z = np.polyfit(data[x_col], data[y_col], 1)
                p = np.poly1d(z)
                ax.plot(data[x_col], p(data[x_col]), "r--", alpha=0.8, linewidth=2)
            
            plt.tight_layout()
            
            return {
                'type': 'scatter_plot',
                'title': f'Scatter Plot: {x_col} vs {y_col}',
                'description': f'Relationship between {x_col} and {y_col} with trend line',
                'image': self._fig_to_base64(fig),
                'columns_analyzed': [x_col, y_col]
            }
            
        except Exception as e:
            logger.error(f"Scatter plot creation failed: {str(e)}")
            return None
        finally:
            plt.close()
    
    def _create_bar_chart(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Create bar chart for categorical data"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Get value counts
            value_counts = df[column].value_counts().head(10)  # Top 10 values
            
            bars = ax.bar(range(len(value_counts)), value_counts.values, color='lightcoral')
            ax.set_xlabel(column)
            ax.set_ylabel('Count')
            ax.set_title(f'Distribution of {column}')
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, value_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                       str(value), ha='center', va='bottom')
            
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return {
                'type': 'bar_chart',
                'title': f'Distribution of {column}',
                'description': f'Bar chart showing frequency distribution of {column}',
                'image': self._fig_to_base64(fig),
                'columns_analyzed': [column]
            }
            
        except Exception as e:
            logger.error(f"Bar chart creation failed: {str(e)}")
            return None
        finally:
            plt.close()
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        try:
            buffer = BytesIO()
            fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            buffer.close()
            return image_base64
            
        except Exception as e:
            logger.error(f"Figure to base64 conversion failed: {str(e)}")
            return ""
