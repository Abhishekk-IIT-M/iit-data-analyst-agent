import logging
from typing import Dict, Any

# We are simplifying the agent. The core logic will be in the LangChainAgent.
# The other classes will be used as "tools" by the LangChainAgent.
from langchain_agent import LangChainAgent

logger = logging.getLogger(__name__)

class DataAnalystAgent:
    """
    Orchestrates the data analysis process by invoking a powerful
    LangChain-based agent that can decide which tools to use.
    """

    def __init__(self):
        """
        Initializes the agent. The main component is the LangChainAgent,
        which will be configured with all the necessary tools.
        """
        # This LangChainAgent will contain the LLM, the tools, and the logic to run them.
        self.langchain_agent = LangChainAgent()
        logger.info("DataAnalystAgent initialized.")

    def run(self, question: str, files: Dict[str, str]) -> Any:
        """
        The new main method that runs the entire analysis.
        It takes the user's question and a dictionary of file paths,
        and passes them to the LangChain agent to get the final answer.

        Args:
            question (str): The content from questions.txt.
            files (Dict[str, str]): A dictionary mapping original filenames to their temporary paths.

        Returns:
            Any: The final result from the LangChain agent, which should be a
                 JSON-serializable dictionary or list.
        """
        logger.info("DataAnalystAgent `run` method started.")

        try:
            # The core of our project is this single call.
            # We are delegating the complex decision-making process to our LangChainAgent.
            # This agent will be responsible for understanding the question, parsing files,
            # scraping websites, running analysis, and generating plots as needed.

            final_answer = self.langchain_agent.execute_task(
                question=question,
                files=files
            )

            logger.info("LangChain agent finished execution.")
            return final_answer

        except Exception as e:
            logger.error(f"An error occurred in the agent's run method: {e}", exc_info=True)
            # Re-raise the exception to be caught by the API endpoint
            raise