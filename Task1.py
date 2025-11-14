import os
from typing import Dict
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import StrOutputParser
from langchain.chains import LLMChain

# Load .env
load_dotenv("C:/Users/ADMIN/Documents/LLMS/.env")

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY missing in .env")

# OpenRouter environment config
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"


class HybridMemoryAssistant:

    def __init__(self, max_tokens: int = 200):

        # LLM
        self.llm = ChatOpenAI(
            model="openai/gpt-4o-mini",
            temperature=0.7,
            max_tokens=max_tokens
        )

        # Memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Output parser
        self.parser = StrOutputParser()

        # LLMChain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            output_parser=self.parser
        )

    def chat(self, user_input: str):
        # Run chain
        response = self.chain.run({
            "chat_history": self.memory.chat_memory.messages,
            "input": user_input
        })

        # Save new message to memory
        self.memory.chat_memory.add_user_message(user_input)
        self.memory.chat_memory.add_ai_message(response)

        return response

    def get_memory_info(self) -> Dict:
        messages = self.memory.chat_memory.messages
        recent = [
            {"type": msg.type, "content": msg.content}
            for msg in messages[-3:]
        ]
        summary = (
            " | ".join([msg.content for msg in messages[:-3]])
            if len(messages) > 3 else ""
        )
        return {
            "recent_messages": recent,
            "token_count": len(str(messages).split()),  # word count approximation
            "summary": summary
        }

    def clear_memory(self):
        self.memory.clear()

    def get_conversation_summary(self) -> str:
        info = self.get_memory_info()
        return info["summary"] if info["summary"] else "No summary yet."


def demo_conversation():
    assistant = HybridMemoryAssistant(max_tokens=150)

    conversation = [
        "Hi! My name is Sarah and I'm a software engineer.",
        "I'm working on a project using Python and React.",
        "I love hiking on weekends and trying new coffee shops.",
        "Can you remind me what I told you about my hobbies?",
        "What programming languages am I using?",
        "I just finished a 10-mile hike yesterday!",
        "Do you remember my name and what I do?",
    ]

    for i, user_input in enumerate(conversation, 1):
        print(f"\nTurn {i}")
        print("User:", user_input)
        resp = assistant.chat(user_input)
        print("Assistant:", resp)

        if i % 3 == 0:
            mem = assistant.get_memory_info()
            print("Memory tokens:", mem["token_count"])
            print("Summary:", mem["summary"])

    print("\nFinal Summary:")
    print(assistant.get_conversation_summary())


if __name__ == "__main__":
    demo_conversation()
