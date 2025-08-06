# Data Analyst Agent

## Overview

A comprehensive Flask-based API application that provides intelligent data analysis capabilities. The system combines statistical analysis, data visualization, web scraping, and AI-powered insights through LangChain integration. It supports multiple data sources including file uploads (CSV, Parquet), web scraping, and S3-stored datasets, making it a versatile tool for data exploration and analysis.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Web Framework
- **Flask-based REST API**: Simple HTTP server with a single main endpoint `/analyze` for processing data analysis requests
- **File Upload Handling**: Supports CSV, TXT, and Parquet files with configurable size limits (100MB max)
- **Template-based Documentation**: HTML interface served through Flask templates for API documentation

### Data Processing Pipeline
- **Modular Agent Architecture**: Main `DataAnalystAgent` class orchestrates multiple specialized components
- **Multi-source Data Support**: Handles file uploads, web scraping, and S3-based datasets through different input channels
- **Statistical Analysis Engine**: Dedicated `StatisticalAnalyzer` for correlation analysis, regression testing, normality tests, and outlier detection
- **DuckDB Integration**: In-memory analytical database for efficient querying of large datasets, particularly S3-stored Parquet files

### Visualization System
- **Matplotlib/Seaborn Backend**: Server-side chart generation using Agg backend for headless environments
- **Base64 Encoding**: Charts are converted to base64 strings for easy JSON API responses
- **Auto-visualization Logic**: Intelligent chart type selection based on data characteristics

### AI Integration Layer
- **LangChain Framework**: Optional integration with OpenAI models for intelligent data insights
- **Pandas DataFrame Agent**: Allows natural language queries against structured data
- **Graceful Degradation**: System continues to function without AI features if API keys are unavailable

### Data Access Patterns
- **Temporary File Management**: Uses system temp directory for uploaded file processing
- **S3 Connectivity**: AWS credentials configured through environment variables
- **Web Scraping**: Trafilatura library for content extraction with BeautifulSoup fallback for table data

### Error Handling Strategy
- **Comprehensive Validation**: File type, size, and format validation before processing
- **Structured Error Responses**: Consistent JSON error format across all endpoints
- **Logging Integration**: Debug-level logging throughout the application for troubleshooting

## External Dependencies

### Core Web Framework
- **Flask**: Main web application framework with file upload capabilities
- **Werkzeug**: HTTP utility library for secure filename handling and request processing

### Data Processing Libraries
- **Pandas**: Primary data manipulation and analysis library
- **NumPy**: Numerical computing foundation for statistical operations
- **DuckDB**: In-memory analytical database with S3 integration capabilities

### Statistical Analysis
- **SciPy**: Statistical functions including correlation tests and probability distributions
- **StatsModels**: Advanced statistical modeling including regression analysis and diagnostic tests

### Visualization
- **Matplotlib**: Core plotting library configured for server environments
- **Seaborn**: Statistical data visualization built on matplotlib

### Web Scraping
- **Trafilatura**: Main text extraction library for web content
- **BeautifulSoup4**: HTML/XML parsing for structured data extraction
- **Requests**: HTTP client for web scraping operations

### AI/ML Integration
- **LangChain**: Framework for LLM integration and agent creation
- **OpenAI API**: Access to GPT models for intelligent analysis (optional)

### Cloud Storage
- **AWS S3**: Object storage integration through DuckDB's httpfs extension
- **AWS Credentials**: Environment variable-based authentication for S3 access

### File Format Support
- **CSV Processing**: Built-in pandas support with encoding detection
- **Parquet Files**: Native support through pandas and DuckDB for efficient columnar data processing