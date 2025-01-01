from pathlib import Path
from src.processor import PDFProcessor, Qdrant
from src.chat import GeminiChat
from src.config import config
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_pdfs(pdfs_path: str, output_path: str):
    processor = PDFProcessor(config())
    qdrant = Qdrant(config())

    image_paths = processor.pdf_to_image(pdfs_path, output_path)
    if not image_paths:
        logger.error("No images were generated from PDFs")
        return

    qdrant.qdrant_setup()
    qdrant.index_documents(image_paths)
    logger.info("PDF processing and indexing complete")


def query_system(query: str):
    qdrant = Qdrant(config())
    chat = GeminiChat(config())

    # Search for relevant images
    points = qdrant.search(query)
    if not points:
        return "No relevant images found", []

    prompt = []
    if points:
        for point in points:
            base64_data = point.payload["base64"]
            prompt.append({"mime_type": "image/jpeg", "data": base64_data})
    prompt.append(query)

    chat.start_chat()
    response = chat.send_message(query)
    return response, points


def main():
    current_path = Path.cwd()
    pdfs_path = list(current_path.glob("**/*.pdf"))
    images_path = current_path / "data" / "images"

    process_pdfs(pdfs_path, images_path)

    query = "your question goes here"

    response, points = query_system(query)
    print(response)


if __name__ == "__main__":
    main()
