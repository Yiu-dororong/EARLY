from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq


class DeepEvalGroqAdapter(DeepEvalBaseLLM):
    def __init__(self, 
                 model_name: str = "llama-3.3-70b-versatile", 
                 temperature: float = 0.0):
        # 1. Instantiate the raw LangChain ChatGroq model internally
        self.chat_model = ChatGroq(
            model=model_name,
            temperature=temperature
        )
        self.model_name = model_name

    def load_model(self):
        return self.chat_model

    # 2. Map DeepEval's synchronous text generator
    def generate(self, prompt: str) -> str:
        response = self.chat_model.invoke([HumanMessage(content=prompt)])
        return str(response.content)

    # 3. Map DeepEval's asynchronous text generator
    async def a_generate(self, prompt: str) -> str:
        response = await self.chat_model.ainvoke([HumanMessage(content=prompt)])
        return str(response.content)

    def get_model_name(self) -> str:
        return f"Groq - {self.model_name}"