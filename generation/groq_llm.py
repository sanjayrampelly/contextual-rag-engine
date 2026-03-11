import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class GroqGenerator:
    def __init__(self, model: str = None):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model or os.getenv("GROQ_MODEL", "llama3-8b-8192")

    def generate(self, query: str, context: str) -> str:
        prompt = f"""
        You are a RAG assistant.

        Rules:
        1) Answer ONLY from the context.
        2) If context is insufficient, say: "I don't know based on the provided documents."
        3) Give a complete answer in 4-6 lines.
        4) Use at least 2 facts from the context.

        QUESTION:
        {query}

        CONTEXT:
        {context}

        ANSWER:
        """


        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()
