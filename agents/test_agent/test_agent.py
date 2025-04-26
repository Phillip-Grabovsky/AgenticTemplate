import utils.LLMClient as LLMClient
from pydantic import BaseModel
from typing import Literal
from prompts.mexico_prompts import Prompts

class CategoryResponse(BaseModel):
        category: Literal["nonessential", "low priority", "medium priority", "high priority", "critical"]

class ClassifierAgent:
    def __init__(self, model="gpt-4.1-nano"):
        self.llm_client = LLM
        self.functions = [
            {
                "name": "classify_text",
                "description": "Classifies a text into a category",
                "parameters": CategoryResponse.schema()
            }
        ]

    def classify(self, text: str) -> str:
        """
        Args:
            text (str): The text to classify

        Returns:
            str: The category of the text
        """
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": text}],
            functions=self.functions,
            function_call={"name": "classify_text"}
        )

        category = response["choices"][0]["message"]["function_call"]["arguments"]
        parsed = CategoryResponse.parse_raw(category)
        return parsed.category