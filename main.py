import asyncio
import logging

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_core import TRACE_LOGGER_NAME
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.langchain import LangChainToolAdapter
from dotenv import load_dotenv
from lib.websearch import google_search

from langchain_community.tools import ShellTool, DuckDuckGoSearchResults

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(TRACE_LOGGER_NAME)
logger.setLevel(logging.DEBUG)

# --------------------------------------------------------------------------------
# Agent Prompts
# --------------------------------------------------------------------------------

planning_agent_prompt = """
You are a planning agent.
Your job is to break down complex tasks into smaller, manageable subtasks, that result in a change in the system.
You should create a complete plan without asking for confirmation.
Your team members are:
    WebSearchAgent: Searches for information
    FixerAgent: Modifies or generates YAML based on gathered information
    RunnerAgent: Applies configurations to a terminal

You only plan and delegate tasks. The WebSearchAgent should search for all the information needed so that the FixerAgent can
craft an API in yaml format that is ready to be applied by the RunnerAgent.

When assigning tasks, use this format:
1. <agent> : <task>

After all tasks are complete, explain the changes that were done in the system and end with "TERMINATE".
"""

web_search_agent_prompt = """
You are a web search agent.
Your only tool is search_tool - use it to find information.
You make only one search call at a time and provide complete information without asking for confirmation.
Once you have the results, respond with them in bullet points if needed.
Do not do calculations based on them. Summaries or short syntheses are ok.
"""

fixer_agent_prompt = """
You are a fixer agent.
You use gathered resources to craft, modify, or correct YAML files.
Make decisions independently and proceed with best practices when information is incomplete.
Clearly indicate any assumptions made in comments.
Your output must always be valid YAML.
"""

runner_agent_prompt = """
You are the runner agent.
You execute commands on a terminal to complete tasks without asking for confirmation.
If asked to apply a YAML, you will run:
    kubectl apply -f <file>
Otherwise, directly execute commands requested by the plan.
Proceed with execution unless there's a clear error condition.
"""

# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------

async def main():
    load_dotenv()

    # Instantiate the LLM
    llm = OpenAIChatCompletionClient(
        model="gpt-4o-2024-08-06",
    )

    # Create the agents
    planning_agent = AssistantAgent(
        name="PlanningAgent",
        description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
        model_client=llm,
        system_message=planning_agent_prompt,
    )

    duckduckSearch = LangChainToolAdapter(DuckDuckGoSearchResults())
    web_search_agent = AssistantAgent(
        name="WebSearchAgent",
        description="A web search agent.",
        tools=[duckduckSearch],
        model_client=llm,
        system_message=web_search_agent_prompt,
    )

    fixer_agent = AssistantAgent(
        name="FixerAgent",
        description="A fixer agent that modifies or generates YAML based on gathered information.",
        model_client=llm,
        system_message=fixer_agent_prompt,
    )

    shell_langchain_tool = LangChainToolAdapter(ShellTool())
    runner_agent = AssistantAgent(
        name="RunnerAgent",
        description="Executes commands on a terminal.",
        model_client=llm,
        system_message=runner_agent_prompt,
        tools=[shell_langchain_tool],
    )

    # Termination conditions
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=30)
    termination = text_mention_termination | max_messages_termination

    team = SelectorGroupChat(
        [planning_agent, web_search_agent, fixer_agent, runner_agent],
        model_client=llm,
        termination_condition=termination,
    )

    # The user’s ask. This is for demo
    # in real it would be a UI, or terminal input
    user_question = """
    This route table has an issue, can you help me resolve it?
    It doesn't configure the gateway for north south traffic

    ```
    apiVersion: networking.istio.io/v1
    kind: VirtualService
    metadata:
      name: reviews-route
      namespace: foo
    spec:
      hosts:
      - reviews # interpreted as reviews.foo.svc.cluster.local
      http:
      - match:
        - uri:
            prefix: "/wpcatalog"
        - uri:
            prefix: "/consumercatalog"
        rewrite:
          uri: "/newcatalog"
    ```

    The answer should be grounded in the official Istio API docs.
    """

    # Stream the conversation in the console
    await Console(team.run_stream(task=user_question))


if __name__ == "__main__":
    asyncio.run(main())
