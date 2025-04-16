from services.LLMClient import LLMClient
from state import State


class ExampleAgent:
    def __init__(self, state: State):
        self.llm_client = LLMClient()
        self.state = state

    async def answer_question(self):
        print("Running example agent")


        response = await self.llm_client.oneShot(
            "sysPrompt2",   #sysPrompt2 is a key in the prompts.json file
            "what is the capital of ",    #if the primpt is not a key in prompts.json, any string input will do
            "gemini-1.5-flash"  #model name from models.json
            data = "france"  #data is a string appended to the user prompt
        )

        self.state.answer = response
        