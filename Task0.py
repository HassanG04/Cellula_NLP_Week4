from assistants import OpenRouterClient, SupportAssistant


def create_customer_support_assistant(client=None):
    return SupportAssistant(client or OpenRouterClient())


if __name__ == "__main__":
    assistant = create_customer_support_assistant()
    print(assistant.invoke({"user_input": input("Customer: ")}).content)
