import os
import logging
import google.generativeai as genai
from .config import config
from .exceptions import ChatError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class GeminiChat:
    """Handles chat interactions with Gemini model."""

    def __init__(self, config: config):
        self.config = config
        self.chat = None

        if not os.getenv("GEMINI_API_KEY"):
            raise ChatError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    def start_chat(self, history=[]):
        """Initialize a new chat session with optional history."""
        try:
            if self.config.system_instruction:
                self.model = genai.GenerativeModel(
                    model_name=self.config.gemini_model,
                    system_instruction=self.config.system_instruction,
                    generation_config=self.config.google_config,
                )
            else:
                self.model = genai.GenerativeModel(
                    model_name=self.config.gemini_model,
                    generation_config=self.config.google_config,
                )

            self.chat = self.model.start_chat(history=history or [])
            logger.info(
                f"Chat session initialized successfully with system prompt \n {self.config.system_instruction}"
            )

        except Exception as e:
            raise ChatError(f"Failed to start chat: {e}")

    def send_message(self, prompt):
        """Send message to chat with optional context from retrieved documents."""
        try:
            if self.chat is None:
                self.start_chat()

            response = self.chat.send_message(prompt)
            return response.text

        except Exception as e:
            raise ChatError(f"Failed to send message: {e}")

    def get_chat_history(self):
        """Retrieve current chat history."""
        if not self.chat:
            logger.warning("No active chat session")
            return []
        return self.chat.history

    def clear_chat(self) -> None:
        """Reset the chat session."""
        self.chat = self.model = None
        logger.info("Chat session cleared")
