import os
import pandas as pd
import logging
import io
import base64
import json
import requests
from typing import Dict, Any, List

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
def analyze_scraped_movie_data(file_path: str) -> str:
    """
    Reads a CSV file of scraped movie data, performs a complete analysis to answer the four
    specific questions (number of $2bn movies before 2000, earliest >$1.5bn film, Rank/Peak
    correlation, and a scatterplot), and returns the final formatted JSON array.
    This is the primary analysis tool for the Wikipedia movie data task.
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
        logger.error(f"Analysis tool failed: {e}", exc_info=True)
        return f"Error during analysis: {e}"

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
        self.tools = [web_scraper, duckdb_sql_querier, analyze_scraped_movie_data]

        # Bind the tools to the model
        llm_with_tools = self.llm.bind_tools(self.tools)

        # Create the new, more efficient prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful data analyst. You must use the provided tools to answer the user's question."),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create the agent
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

        # The new agent structure doesn't use the 'files' variable in the prompt,
        # but we keep it in the function signature for consistency.
        response = self.agent_executor.invoke({ "input": question })

        # The output from a tool-calling agent is often the final answer directly
        # If the last step was a tool call that produced the final JSON, we extract it.
        final_output = response.get('output', '')
        try:
            # Check if the output is already a JSON string from our analysis tool
            return json.loads(final_output)
        except (json.JSONDecodeError, TypeError):
            return final_output