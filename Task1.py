from assistants import HybridMemoryAssistant

if __name__ == "__main__":
    assistant = HybridMemoryAssistant()
    while True:
        text = input("You (quit to exit): ")
        if text.strip().lower() == "quit":
            break
        print(assistant.chat(text))
