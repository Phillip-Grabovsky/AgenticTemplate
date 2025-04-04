from services.LLMClient import LLMClient

class State:
    def __init__(self):
        self.question: str
        self.answer: str
        self.llm_client = LLMClient()
        