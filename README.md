# LLM Client

A Python client for interacting with multiple LLM providers through their APIs.

## Supported Providers
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Google (Gemini)
- Perplexity (Sonar)
- Groq
- NVIDIA (Nemotron)

## Setup

1. Install dependencies:
   ```
   pip install openai python-dotenv google-generativeai groq
   ```
2. Create a `.env` file with your API keys:
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

#### oneShot()
```python
async def oneShot(
    sysPrompt: str,      # System prompt or key from prompts.json
    usrPrompt: str,      # User prompt or key from prompts.json
    model: str,          # Model nickname from models.json
    data: str = ""       # Additional text appended to user prompt
) -> str
```

#### conversation()
```python
async def conversation(
    convoId: str,        # Unique identifier for the conversation
    message: str,        # Message or key from prompts.json
    model: str,          # Model nickname from models.json
    sysPrompt: str = "", # Optional system prompt
    data: str = ""       # Additional text appended to user prompt
) -> str
```

### Model Configuration

Configure models in `services/models.json`:
```json
{
    "model-nickname": [
        "actual-model-name",  # Provider's model name
        "provider-id",        # "openAI", "anthropic", "google", "groq", etc.
        "base-url"           # Required for OpenAI-compatible APIs (e.g., Groq)
    ]
}
```

Example configurations:
```json
{
    "gpt-4": ["gpt-4", "openAI", ""],
    "gemini-2.0-flash" : ["gemini-2.0-flash", "google", ""],
    "llama-3.3-70b": ["llama-3.3-70b-versatile", "groq", "https://api.groq.com/openai/v1"]
}
```

### Project Structure
- `flow.py`: Main execution flow
- `state.py`: Shared whiteboard for agents to contribute to &  draw from
- `agents/`: Individual agent implementations

### Example Usage

```python
from services.LLMClient import LLMClient
from state import State

state = State()
client = LLMClient()

# Single-turn with custom data
response = await client.oneShot(
    "sysPrompt1",           # From prompts.json
    "basePrompt",          # From prompts.json
    "claude-3.7-sonnet",    # From models.json
    data="in New York"          # Appended to basePrompt
)

# Multi-turn conversation
response = await client.conversation(
    "weather_chat",              #convo ID
    "basePrompt",               #from prompts.json
    "claude-3.7-sonnet",        #model nickname
    "You are a weather expert", #regular strings also work in place of prompt keys from json
    data="in New York"               #data is appended to the user prompt
)
```


