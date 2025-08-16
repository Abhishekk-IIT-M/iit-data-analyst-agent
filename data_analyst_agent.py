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
    The main method that runs the analysis by invoking the LangChain agent.
    """
    logger.info("DataAnalystAgent `run` method started.")

    try:
        # CORRECTED: The LangChain agent's main function is now invoke() on the agent_executor
        # We will build the input and pass it to the agent executor.

        # We need to inform the agent about the files it has access to.
        if files:
            file_names = ", ".join(files.keys())
            question_with_context = f"{question}\n\nThe user has provided the following file(s): {file_names}. Use the most relevant file for the analysis."
        else:
            question_with_context = question

        # Directly call the agent executor
        response = self.langchain_agent.agent_executor.invoke({
            "input": question_with_context
        })

        final_output = response.get('output', '')
        logger.info("LangChain agent finished execution.")

        try:
            return json.loads(final_output)
        except (json.JSONDecodeError, TypeError):
            return final_output

    except Exception as e:
        logger.error(f"An error occurred in the agent's run method: {e}", exc_info=True)
        raise