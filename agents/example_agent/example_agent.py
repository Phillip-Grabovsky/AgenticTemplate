from services.LLMClient import LLMClient
from state import State
from agents import prompts
from agents.pydantic_models import (
    ExampleAgentResponse
)

class ExampleAgent:
    def __init__(self, state: State):
        self.state = state
        self.llm_client = state.llm_client


    async def answer_question(self):
        print("Running example agent")


        response = await self.llm_client.oneShot(
            prompts.SYSPROMPT2,   #prompts can be stored in a file
            "what is the capital of ",    #if the primpt is not a key in prompts.json, any string input will do
            "gemini-1.5-flash"  #model name from models.json
            data = "france"  #data is a string appended to the user prompt
        )

        response = self.llm.validate_response(response, ExampleAgentResponse)

        self.state.answer = response
        