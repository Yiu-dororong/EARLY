import os

from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


class DeepEvalGoogleAdapter(DeepEvalBaseLLM):
    def __init__(self,
                 model_name: str = "gemini-3.8-flash",
                 temperature: float = 0.0):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.chat_model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
        )
        self.model_name = model_name

    def load_model(self):
        return self.chat_model

    def generate(self, prompt: str) -> str:
        response = self.chat_model.invoke([HumanMessage(content=prompt)])
        return str(response.content)

    async def a_generate(self, prompt: str) -> str:
        response = await self.chat_model.ainvoke([HumanMessage(content=prompt)])
        return str(response.content)

    def get_model_name(self) -> str:
        return f"Google - {self.model_name}"
