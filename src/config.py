from dataclasses import dataclass


@dataclass
class config:
    collection_name: str = "vision_rag"
    image_model: str = "jinaai/jina-clip-v2"
    gemini_model: str = "gemini-1.5-flash-002"
    qdrant_url: str = (
        "your qdrant cloud url"
    )
    system_instruction: str = """You are a helpful RAG assistant.
    When user asks you a question, answers only based on the image given.
    If you cant answer based on the image or no image is given, reply 'I dont know'.
    Provide all necessary information.""".strip()
    google_config = {"temperature": 0}
    colpali_model: str = "vidore/colqwen2-v1.0"
    cpu_threads: int = 2
    batch_size: int = 2
    image_seq_length: int = 1024  # Exact sequence length from ColPali config
    grid_size: int = 32  # sqrt(1024) for grid reshaping
    max_image_patches: int = 768  # Max patches from ColQwen2 documentation
