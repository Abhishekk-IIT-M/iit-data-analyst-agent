import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr, chi2_contingency
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_white
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis functionality
    """
    
    def __init__(self):
        pass
    
    def analyze(self, df: pd.DataFrame, analysis_type: str = 'basic', tests: List[str] = None) -> Dict[str, Any]:
        """
        Main statistical analysis method
        """
        try:
            results = {}
            
            # Basic descriptive statistics
            results['descriptive_stats'] = self._get_descriptive_stats(df)
            
            # Correlation analysis
            results['correlation_analysis'] = self._correlation_analysis(df)
            
            if analysis_type in ['comprehensive', 'statistical']:
                # Advanced statistical tests
                results['normality_tests'] = self._test_normality(df)
                results['outlier_analysis'] = self._detect_outliers(df)
                
                # Regression analysis if applicable
                if len(self._get_numeric_columns(df)) >= 2:
                    results['regression_analysis'] = self._regression_analysis(df)
                
                # Custom statistical tests
                if tests:
                    results['custom_tests'] = self._run_custom_tests(df, tests)
            
            return results
            
        except Exception as e:
            logger.error(f"Statistical analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _get_descriptive_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive descriptive statistics"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                return {'message': 'No numeric columns found for descriptive statistics'}
            
            desc_stats = numeric_df.describe()
            
            # Additional statistics
            additional_stats = {}
            for col in numeric_df.columns:
                series = numeric_df[col].dropna()
                if len(series) > 0:
                    additional_stats[col] = {
                        'variance': float(series.var()),
                        'skewness': float(series.skew()),
                        'kurtosis': float(series.kurtosis()),
                        'mode': float(series.mode().iloc[0]) if len(series.mode()) > 0 else None,
                        'range': float(series.max() - series.min()),
                        'iqr': float(series.quantile(0.75) - series.quantile(0.25))
                    }
            
            return {
                'basic_stats': desc_stats.to_dict(),
                'additional_stats': additional_stats
            }
            
        except Exception as e:
            logger.error(f"Error in descriptive statistics: {str(e)}")
            return {'error': str(e)}
    
    def _correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive correlation analysis"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.shape[1] < 2:
                return {'message': 'Need at least 2 numeric columns for correlation analysis'}
            
            # Pearson correlation
            pearson_corr = numeric_df.corr(method='pearson')
            
            # Spearman correlation
            spearman_corr = numeric_df.corr(method='spearman')
            
            # Significant correlations (|r| > 0.5)
            strong_correlations = []
            for i in range(len(pearson_corr.columns)):
                for j in range(i+1, len(pearson_corr.columns)):
                    col1 = pearson_corr.columns[i]
                    col2 = pearson_corr.columns[j]
                    r_value = pearson_corr.iloc[i, j]
                    
                    if abs(r_value) > 0.5:
                        strong_correlations.append({
                            'variable1': col1,
                            'variable2': col2,
                            'pearson_r': float(r_value),
                            'strength': 'strong' if abs(r_value) > 0.7 else 'moderate'
                        })
            
            return {
                'pearson_correlation': pearson_corr.to_dict(),
                'spearman_correlation': spearman_corr.to_dict(),
                'strong_correlations': strong_correlations
            }
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            return {'error': str(e)}
    
    def _test_normality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Test normality of numeric variables"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            normality_results = {}
            
            for col in numeric_df.columns:
                series = numeric_df[col].dropna()
                if len(series) >= 8:  # Minimum sample size for normality tests
                    # Shapiro-Wilk test
                    shapiro_stat, shapiro_p = stats.shapiro(series)
                    
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_p = stats.kstest(series, 'norm', args=(series.mean(), series.std()))
                    
                    normality_results[col] = {
                        'shapiro_wilk': {
                            'statistic': float(shapiro_stat),
                            'p_value': float(shapiro_p),
                            'is_normal': shapiro_p > 0.05
                        },
                        'kolmogorov_smirnov': {
                            'statistic': float(ks_stat),
                            'p_value': float(ks_p),
                            'is_normal': ks_p > 0.05
                        }
                    }
            
            return normality_results
            
        except Exception as e:
            logger.error(f"Error in normality testing: {str(e)}")
            return {'error': str(e)}
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using multiple methods"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            outlier_results = {}
            
            for col in numeric_df.columns:
                series = numeric_df[col].dropna()
                if len(series) > 0:
                    # IQR method
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]
                    
                    # Z-score method
                    z_scores = np.abs(stats.zscore(series))
                    z_outliers = series[z_scores > 3]
                    
                    outlier_results[col] = {
                        'iqr_method': {
                            'count': len(iqr_outliers),
                            'percentage': float(len(iqr_outliers) / len(series) * 100),
                            'values': iqr_outliers.tolist()
                        },
                        'z_score_method': {
                            'count': len(z_outliers),
                            'percentage': float(len(z_outliers) / len(series) * 100),
                            'values': z_outliers.tolist()
                        }
                    }
            
            return outlier_results
            
        except Exception as e:
            logger.error(f"Error in outlier detection: {str(e)}")
            return {'error': str(e)}
    
    def _regression_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform regression analysis"""
        try:
            numeric_df = df.select_dtypes(include=[np.number]).dropna()
            
            if numeric_df.shape[1] < 2:
                return {'message': 'Need at least 2 numeric columns for regression analysis'}
            
            # Use first column as dependent variable, rest as independent
            y = numeric_df.iloc[:, 0]
            X = numeric_df.iloc[:, 1:]
            
            # Add constant for intercept
            X_with_const = sm.add_constant(X)
            
            # Fit OLS model
            model = sm.OLS(y, X_with_const).fit()
            
            # Calculate VIF for multicollinearity
            vif_data = []
            if X.shape[1] > 1:
                for i in range(X.shape[1]):
                    vif = variance_inflation_factor(X.values, i)
                    vif_data.append({
                        'variable': X.columns[i],
                        'vif': float(vif) if not np.isnan(vif) else None
                    })
            
            return {
                'model_summary': {
                    'r_squared': float(model.rsquared),
                    'adj_r_squared': float(model.rsquared_adj),
                    'f_statistic': float(model.fvalue),
                    'f_pvalue': float(model.f_pvalue),
                    'aic': float(model.aic),
                    'bic': float(model.bic)
                },
                'coefficients': {
                    var: {
                        'coef': float(model.params[var]),
                        'std_err': float(model.bse[var]),
                        't_value': float(model.tvalues[var]),
                        'p_value': float(model.pvalues[var])
                    } for var in model.params.index
                },
                'multicollinearity': vif_data,
                'dependent_variable': y.name,
                'independent_variables': list(X.columns)
            }
            
        except Exception as e:
            logger.error(f"Error in regression analysis: {str(e)}")
            return {'error': str(e)}
    
    def _run_custom_tests(self, df: pd.DataFrame, tests: List[str]) -> Dict[str, Any]:
        """Run custom statistical tests"""
        try:
            results = {}
            numeric_df = df.select_dtypes(include=[np.number])
            
            for test in tests:
                if test.lower() == 't_test' and numeric_df.shape[1] >= 2:
                    # Perform t-test between first two numeric columns
                    col1, col2 = numeric_df.columns[:2]
                    stat, p_val = stats.ttest_ind(numeric_df[col1].dropna(), numeric_df[col2].dropna())
                    results['t_test'] = {
                        'statistic': float(stat),
                        'p_value': float(p_val),
                        'variables': [col1, col2]
                    }
                
                elif test.lower() == 'chi_square':
                    # Chi-square test for categorical variables
                    cat_cols = df.select_dtypes(include=['object', 'category']).columns
                    if len(cat_cols) >= 2:
                        contingency_table = pd.crosstab(df[cat_cols[0]], df[cat_cols[1]])
                        chi2, p_val, dof, expected = chi2_contingency(contingency_table)
                        results['chi_square'] = {
                            'statistic': float(chi2),
                            'p_value': float(p_val),
                            'degrees_of_freedom': int(dof),
                            'variables': list(cat_cols[:2])
                        }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in custom tests: {str(e)}")
            return {'error': str(e)}
    
    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of numeric column names"""
        return df.select_dtypes(include=[np.number]).columns.tolist()
