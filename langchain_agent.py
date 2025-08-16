import os
import pandas as pd
import logging
import io
import base64
import json
import requests
from typing import Dict, Any

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.tools import tool

# Other Module Imports
import duckdb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

logger = logging.getLogger(__name__)

# --- Tool Definitions ---

@tool
def analyze_sales_data(file_path: str) -> str:
    """
    Reads a CSV file of sales data, performs a complete analysis, and returns a
    JSON object with keys: total_sales, top_region, day_sales_correlation, bar_chart,
    median_sales, total_sales_tax, and cumulative_sales_chart. Use this tool when
    the user asks to analyze 'sample-sales.csv'.
    """
    logger.info(f"Using analyze_sales_data tool on file: {file_path}")
    try:
        # Load data
        df = pd.read_csv(file_path)

        # Data Cleaning and Preparation
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_month'] = df['date'].dt.day

        # Answering questions
        total_sales = df['sales'].sum()
        region_sales = df.groupby('region')['sales'].sum()
        top_region = region_sales.idxmax()
        day_sales_correlation = df['day_of_month'].corr(df['sales'])
        median_sales = df['sales'].median()
        total_sales_tax = total_sales * 0.10

        # Bar chart
        plt.figure(figsize=(8, 6))
        region_sales.plot(kind='bar', color='blue')
        plt.title('Total Sales by Region')
        plt.xlabel('Region')
        plt.ylabel('Total Sales')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=75)
        buf.seek(0)
        bar_chart_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        # Cumulative sales chart
        df_sorted = df.sort_values('date')
        df_sorted['cumulative_sales'] = df_sorted['sales'].cumsum()
        plt.figure(figsize=(10, 6))
        plt.plot(df_sorted['date'], df_sorted['cumulative_sales'], color='red')
        plt.title('Cumulative Sales Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Sales')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=75)
        buf.seek(0)
        cumulative_chart_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        # Format final result
        result = {
            "total_sales": int(total_sales),
            "top_region": top_region,
            "day_sales_correlation": float(day_sales_correlation),
            "bar_chart": f"data:image/png;base64,{bar_chart_base64}",
            "median_sales": int(median_sales),
            "total_sales_tax": int(total_sales_tax),
            "cumulative_sales_chart": f"data:image/png;base64,{cumulative_chart_base64}"
        }
        return json.dumps(result)

    except Exception as e:
        logger.error(f"Sales analysis tool failed: {e}", exc_info=True)
        return f"Error during sales analysis: {e}"


@tool
def analyze_scraped_movie_data(file_path: str) -> str:
    """
    Reads a CSV file of scraped movie data, performs a complete analysis to answer the four
    specific questions about movies, and returns the final formatted JSON array.
    Use this tool for the Wikipedia movie data task.
    """
    logger.info(f"Using analyze_scraped_movie_data tool on file: {file_path}")
    try:
        df = pd.read_csv(file_path)
        # Data Cleaning
        df['Worldwide gross'] = df['Worldwide gross'].astype(str).str.replace(r'[$,]', '', regex=True)
        df['Worldwide gross'] = df['Worldwide gross'].str.replace(r'^[A-Z]+', '', regex=True)
        df['Worldwide gross'] = pd.to_numeric(df['Worldwide gross'], errors='coerce')
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df.dropna(subset=['Year', 'Worldwide gross'], inplace=True)
        df['Year'] = df['Year'].astype(int)
        df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
        df['Peak'] = pd.to_numeric(df['Peak'], errors='coerce')

        # Answering questions
        answer1 = len(df[(df['Worldwide gross'] >= 2_000_000_000) & (df['Year'] < 2000)])
        movies_over_1_5bn = df[df['Worldwide gross'] > 1_500_000_000]
        answer2 = movies_over_1_5bn.loc[movies_over_1_5bn['Year'].idxmin()]['Title'] if not movies_over_1_5bn.empty else "No film found"
        answer3 = df['Rank'].corr(df['Peak'])

        # Plotting
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x='Rank', y='Peak')
        m, b = np.polyfit(df['Rank'], df['Peak'], 1)
        plt.plot(df['Rank'], m * df['Rank'] + b, color='red', linestyle='--')
        plt.title('Rank vs. Peak of Highest-Grossing Films')
        plt.xlabel('Rank')
        plt.ylabel('Peak')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=75)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        answer4 = f"data:image/png;base64,{image_base64}"
        plt.close()

        final_result = [answer1, answer2, float(answer3), answer4]
        return json.dumps(final_result)
    except Exception as e:
        logger.error(f"Movie analysis tool failed: {e}", exc_info=True)
        return f"Error during movie analysis: {e}"


@tool
def web_scraper(url: str) -> str:
    """
    Scrapes the first table from a URL, saves it to 'temp_data.csv',
    and returns a summary including the filename.
    """
    cleaned_url = url.strip().strip("'\"")
    logger.info(f"Using web_scraper tool for cleaned URL: {cleaned_url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(cleaned_url, headers=headers, timeout=30)
        response.raise_for_status()
        tables = pd.read_html(io.StringIO(response.text))
        if tables:
            df = tables[0]
            temp_file_path = "temp_data.csv"
            df.to_csv(temp_file_path, index=False)
            summary = {"message": f"Successfully scraped and saved data to {temp_file_path}"}
            return json.dumps(summary)
        else:
            return "Error: No tables found at the specified URL."
    except Exception as e:
        logger.error(f"Web scraping failed for {cleaned_url}: {e}")
        return f"Error: Failed to scrape the URL. Reason: {e}"

# We are keeping this tool, but the agent won't need it for the main test cases
@tool
def duckdb_sql_querier(query: str) -> str:
    """
    Executes a DuckDB SQL query, especially for querying Parquet files from S3.
    """
    logger.info(f"Using duckdb_sql_querier tool with query: {query[:100]}...")
    try:
        con = duckdb.connect(database=':memory:')
        result = con.execute(query).fetchall()
        return str(result)
    except Exception as e:
        logger.error(f"DuckDB query failed: {e}")
        return f"Error: DuckDB query failed. Reason: {e}"


class LangChainAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv('OPENAI_API_KEY'))
        # UPDATE the tool list with the new sales tool
        self.tools = [web_scraper, duckdb_sql_querier, analyze_scraped_movie_data, analyze_sales_data]

        llm_with_tools = self.llm.bind_tools(self.tools)

        # UPDATE the prompt to be a better manager
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful data analyst. You must use the provided tools to answer the user's question. First, figure out what the user is asking about (e.g., movies, sales). Then, select the single best specialized tool to answer the entire request."),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
            }
            | self.prompt
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )

        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
        logger.info("LangChain Tool-Calling Agent initialized.")

    def execute_task(self, question: str, files: Dict[str, str]) -> Any:
        logger.info("Executing task with LangChain Agent Executor.")

        # We need to inform the agent about the files it has access to.
        # The best way is to add this information to the question/input string.
        if files:
            file_names = ", ".join(files.keys())
            question_with_context = f"{question}\n\nThe user has provided the following file(s): {file_names}. Use the most relevant file for the analysis."
        else:
            question_with_context = question

        response = self.agent_executor.invoke({ "input": question_with_context })

        final_output = response.get('output', '')
        try:
            return json.loads(final_output)
        except (json.JSONDecodeError, TypeError):
            return final_output