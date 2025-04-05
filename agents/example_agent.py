from services.LLMClient import LLMClient
from state import State


class ExampleAgent:
    def __init__(self, state: State):
        self.llm_client = LLMClient()
        self.state = state

    async def answer_question(self):
        print("Running example agent")
        
        response = await self.llm_client.oneShot(   #use conversation() to store conversation history under a nickname
            "sysPrompt2",   #If your prompt is in the prompts.json file, you can use its key.
            self.state.question,    #If your prompt is not, any string input will do.
            "claude-3.7-sonnet"  #The model nicknames are in models.json.
        )

        self.state.answer = response
        
