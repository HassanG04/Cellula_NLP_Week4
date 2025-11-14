import os
from dotenv import load_dotenv

# langchain APIs vary between releases; import defensively and provide
# a simple fallback so this script can run in environments without
# the newer langchain modules installed.
try:
    from langchain.chat_models import ChatOpenAI
except Exception:
    ChatOpenAI = None

try:
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
except Exception:
    ChatPromptTemplate = None
    MessagesPlaceholder = None

# Load .env
load_dotenv("C:/Users/ADMIN/Documents/LLMS/.env")

# Get OpenRouter API key from .env
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY missing in .env")

# Set environment variables for OpenRouter
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load .env
load_dotenv("C:/Users/ADMIN/Documents/LLMS/.env")

# Get OpenRouter API key from .env
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY missing in .env")

# Set environment variables for OpenRouter
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"


def create_customer_support_assistant():
    """
    Creates a customer support assistant with defined role and boundaries.
    """

    system_prompt = """You are a professional customer support representative for TechCorp, 
a software company specializing in productivity tools.

Your Role and Responsibilities:
- Assist customers with technical issues, billing questions, and general inquiries
- Maintain a friendly, patient, and professional tone at all times
- Provide clear, step-by-step solutions to technical problems
- Escalate complex issues that require engineering involvement

Your Boundaries:
- DO NOT make promises about features that don't exist
- DO NOT provide refunds without supervisor approval
- DO NOT share internal company information or roadmaps
- DO NOT engage in discussions outside of customer support topics

Your Style:
- Use clear, jargon-free language unless the customer demonstrates technical expertise
- Show empathy for customer frustrations
- Keep responses concise but thorough
- Always end with asking if there's anything else you can help with

Company Policies to Remember:
- Standard refund window is 30 days
- Technical support is available for all paid tiers
- Free tier users have community support only
"""

    # If langchain prompt/LLM implementations aren't available in this
    # environment, return a simple fallback assistant that echoes input.
    if ChatPromptTemplate is None or ChatOpenAI is None:
        from types import SimpleNamespace

        class FallbackAssistant:
            def invoke(self, inputs):
                user_text = inputs.get("user_input") if isinstance(inputs, dict) else str(inputs)
                return SimpleNamespace(content=(
                    "[fallback assistant] LangChain prompt/LLM not available. "
                    f"Received: {user_text}"
                ))

        return FallbackAssistant()

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="user_input")
    ])

    # LLM using OpenRouter
    llm = ChatOpenAI(
        model="openai/gpt-3.5-turbo",  # OpenRouter compatible model
        temperature=0.7
    )

    # Combine prompt + LLM
    chain = prompt | llm

    return chain


def main():
    """
    Example usage of the customer support assistant
    """

    # Create the assistant
    assistant = create_customer_support_assistant()

    # Test with different customer queries
    test_queries = [
        "I want a refund for my subscription",
        "How do I reset my password?",
        "Can you tell me about upcoming features?",
        "My app keeps crashing when I try to export files"
    ]

    print("=" * 70)
    print("CUSTOMER SUPPORT ASSISTANT - Testing System Prompt")
    print("=" * 70)

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Customer Query {i} ---")
        print(f"Customer: {query}")

        try:
            # Get response from the assistant
            response = assistant.invoke({"user_input": query})
            print(f"\nAssistant: {response.content}")
        except Exception as e:
            print(f"Error: {e}")

        print("\n" + "-" * 70)


if __name__ == "__main__":
    main()
