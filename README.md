# Multi-Agent System that connects Autogen, Langchain, and MCP

This project demonstrates a multi-agent system using Autogen that assists with Kubernetes configuration tasks by combining
popular frameworks such as Autogen, langchain and shortly to be added too MCP.

**The agents collaborate to research, modify, and apply YAML configurations.**

## Agents

The system comprises four specialized agents:

- **PlanningAgent:** Decomposes complex tasks into smaller subtasks and delegates them to other agents.
- **WebSearchAgent:** Gathers information from the web using DuckDuckGo search.
- **FixerAgent:** Creates, modifies, or corrects YAML files based on gathered information and best practices.
- **RunnerAgent:** Executes commands on the terminal, such as applying YAML configurations using `kubectl`.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/rinormaloku/autogen-langchain-mcp-mix.git
    cd autogen-langchain-mcp-mix
    ```

2. **Install dependencies in a `venv`:**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

## Configuration

1. **Set up environment variables:**

    Copy the values from `.env.example` to a new file named `.env`:

    ```
    cp .env.example .env
    ```

    Update the environment variables in the `.env` file as needed.


## Usage

1. **Run the main script:**

    ```bash
    python main.py
    ```

    This will start the multi-agent system and initiate a conversation based on the predefined `user_question` in `main.py`. The agents will collaborate to address the issue related to the Istio VirtualService configuration.


## Notes

- This project is a demonstration and may require adjustments for real-world applications.
- The `RunnerAgent`'s ability to execute commands depends on the environment and permissions.
- The termination condition is based on a maximum message count (30) or the mention of the word "TERMINATE".
- Ensure that you have `kubectl` configured correctly if you intend to apply the generated YAML configurations.
