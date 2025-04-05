# LLM Client

A flexible and extensible Python client for interacting with LLMs and inference providers through their respective APIs.

## Overview

This project provides a unified interface to interact with multiple LLM providers including:
- OpenAI (GPT-4o, GPT-3.5, etc.)
- Anthropic (Claude models)
- Google (Gemini models)
- Perplexity (Sonar models)
- Groq
- NVIDIA (Nemotron)

The client handles API authentication, conversation management, and request formatting for each provider, allowing you to focus on the content rather than implementation details.

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install openai python-dotenv google-generativeai groq
   ```
3. Create a `.env` file in the root directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   PPLX_API_KEY=your_perplexity_key
   ANTHROPIC_API_KEY=your_anthropic_key
   GOOGLE_API_KEY=your_google_key
   GROQ_API_KEY=your_groq_key
   NVIDIA_API_KEY=your_nvidia_key
   ```

## Usage

### Core Functions

The LLMClient provides two main methods for interacting with LLMs:

#### oneShot()
```python
async def oneShot(
    sysPrompt: str,      # System prompt or key from prompts.json
    usrPrompt: str,      # User prompt or key from prompts.json
    model: str,          # Model nickname from models.json
    data: str = "",      # Optional additional data
    use_groq: bool = False  # Optional flag to use Groq
) -> str
```
Use this for single-turn conversations where you don't need to maintain conversation history.

#### conversation()
```python
async def conversation(
    convoId: str,        # Unique identifier for the conversation
    message: str,        # Message or key from prompts.json
    model: str,          # Model nickname from models.json
    sysPrompt: str = "", # Optional system prompt
    data: str = "",      # Optional additional data
    use_groq: bool = False  # Optional flag to use Groq
) -> str
```
Use this for multi-turn conversations where you want to maintain context between messages.

### Project Structure

The project is organized into several key components:

#### `flow.py`
- Orchestrates the overall execution flow
- Manages the lifecycle of agents and state

#### `state.py`
- Stores conversation history, user inputs, and agent outputs
- Acts as a shared whiteboard for agents working together on a problem

#### `agents/` directory
- Contains individual agent implementations
- Each agent is responsible for specific tasks or domains
- Agents can be easily added or modified without changing the core flow

### Example Usage

```python
from services.LLMClient import LLMClient

# Initialize client
client = LLMClient()

# Single-turn conversation
response = await client.oneShot(
    "sysPrompt1",           # System prompt key
    "What is the weather?", # User question
    "claude-3.7-sonnet"     # Model to use
)

# Multi-turn conversation
response = await client.conversation(
    "weather_chat",         # Conversation ID
    "What's the forecast?", # User message
    "claude-3.7-sonnet",    # Model to use
)
```


