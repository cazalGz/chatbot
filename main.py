from src.console_chat import ConsoleChat
from src.config import MAX_HISTORY

def main():
    # Crear el chat con historial limitado
    chat = ConsoleChat(max_history=MAX_HISTORY)
    chat.chat()

if __name__ == "__main__":
    main()
