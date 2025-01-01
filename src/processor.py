import torch
import fitz
import logging
import base64
import zlib
from pathlib import Path
import PIL.Image as im
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from colpali_engine.models import ColQwen2, ColQwen2Processor
from tqdm import tqdm
import os
from .config import config
from .exceptions import ProcessingError
from .utils import find_factor_pair

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)


class PDFProcessor:
    def __init__(self, config: config):
        self.config = config

    def pdf_to_image(self, pdf_path, output_path, max_size=1024):
        """Convert PDF pages to images with controlled dimensions."""
        try:
            output_path = Path(output_path)
            output_path.mkdir(exist_ok=True)

            if not pdf_path or len(pdf_path) == 0:
                print(f"No PDF files found in {pdf_path}")
                return None

            all_images = []

            for pdf in pdf_path:
                logger.info(f"Processing {pdf.name}")
                pdf_output = output_path / pdf.stem
                pdf_output.mkdir(exist_ok=True)

                doc = fitz.open(pdf)
                images = []

                for page_number in range(doc.page_count):
                    page = doc[page_number]
                    rect = page.rect

                    # Calculate scaling factor to maintain aspect ratio
                    width = rect.width
                    height = rect.height
                    scale = min(max_size / width, max_size / height)

                    # Create matrix with calculated scale
                    matrix = fitz.Matrix(scale, scale)

                    pixels = page.get_pixmap(matrix=matrix)
                    image_path = (
                        pdf_output / f"{pdf.stem.replace(' ','_')}_page_{page_number + 1}.jpg"
                    )
                    pixels.save(str(image_path))
                    images.append(image_path)

                doc.close()
                all_images.extend(images)
                print(f"Processed {len(images)} images from {pdf.name}")

            return all_images

        except Exception as e:
            raise ProcessingError(f"Error converting PDFs: {str(e)}")


class Qdrant:
    def __init__(self, config: config):
        self.config = config
        self.qdrant_client = QdrantClient(
            url=self.config.qdrant_url, api_key=os.getenv("QDRANT_API")
        )
        self.model = SentenceTransformer(
            self.config.image_model, trust_remote_code=True, truncate_dim=768
        )
        self.model.parallel_tokenization = True
        self.model.parallel_tokenization_threads = self.config.cpu_threads
        torch.set_num_threads(self.config.cpu_threads)
        self.colpali = ColQwen2.from_pretrained(
            self.config.colpali_model,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        ).eval()
        self.colpali_processor = ColQwen2Processor.from_pretrained(self.config.colpali_model)

    def qdrant_setup(self):
        if not self.qdrant_client.collection_exists(self.config.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.config.collection_name,
                on_disk_payload=True,
                vectors_config={
                    "dense": models.VectorParams(
                        size=768,
                        distance=models.Distance.COSINE,
                        on_disk=True,
                    ),
                    "mean_pooling": models.VectorParams(
                        size=128,
                        distance=models.Distance.COSINE,
                        on_disk=True,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        ),
                        quantization_config=models.BinaryQuantization(
                            binary=models.BinaryQuantizationConfig(always_ram=False)
                        ),
                    ),
                },
            )

    def get_image_embeddings(self, image_paths=None, prompt=None):
        if image_paths:
            images = [im.open(path) for path in image_paths]
            return self.model.encode(
                images, normalize_embeddings=True, batch_size=self.config.batch_size
            )
        elif prompt:
            return self.model.encode(prompt, normalize_embeddings=True)

    def get_colpali_embeddings(self, image_paths):
        try:
            images = [im.open(path) for path in image_paths]
        except Exception as e:
            raise ProcessingError(f"Error loading images: {str(e)}")

        batch_images = self.colpali_processor.process_images(images)

        with torch.no_grad():
            image_embeddings = self.colpali(**batch_images)

            batch_size = len(images)
            seq_length = (
                image_embeddings.shape[1] - 6
            )  # Subtract special tokens <bos>Describe the image.
            embed_dim = image_embeddings.shape[2]

            special_tokens = image_embeddings[:, -6:, :]  # Last 6 tokens
            main_sequence = image_embeddings[:, :seq_length, :]
            grid_size = int(seq_length**0.5)

            print("batch_size:", batch_size)
            print("seq_length:", seq_length)
            print("embed_dim:", embed_dim)
            print("special_tokens.shape:", special_tokens.shape)
            print("main_sequence.shape:", main_sequence.shape)
            print("grid_size:", grid_size)

            if grid_size * grid_size == seq_length:
                # Reshape for perfect squares
                reshaped = main_sequence.reshape((batch_size, grid_size, grid_size, embed_dim))
            else:
                grid_h, grid_w = find_factor_pair(seq_length)
                print("Dynamic factor pair for seq_length =", (grid_h, grid_w))
                reshaped = main_sequence.reshape((batch_size, grid_h, grid_w, embed_dim))

            # Pool and concatenate
            try:
                if grid_h < grid_w:
                    mean_pooled = torch.mean(reshaped, dim=1)
                else:
                    mean_pooled = torch.mean(reshaped, dim=2)
                final_embeddings = torch.cat((mean_pooled, special_tokens), dim=1)

                logger.info(f"Processed batch with sequence length {seq_length}")
                print(f"processed {', '.join([path.name for path in image_paths])}")

                return final_embeddings.cpu().float().numpy().tolist()

            except RuntimeError as e:
                print(f"Reshape failed with error: {str(e)}")
                logger.error(f"Failed to reshape tensor of size {main_sequence.shape}")
                raise ProcessingError(f"Embedding reshape failed: {str(e)}")

    def colpali_query(self, query):
        with torch.no_grad():
            batch_query = self.colpali_processor.process_queries([query])
            mask_without_pad = batch_query.input_ids.bool().unsqueeze(-1)

        query_embedding = self.colpali(**batch_query)
        query_without_pad = torch.masked_select(query_embedding, mask_without_pad).view(
            1, -1, 128
        )  # without <pad> tokens
        return query_without_pad[0].cpu().float().numpy().tolist()

    def search(self, query):
        image_embedding = self.get_image_embeddings(prompt=query)
        colpali_embedding = self.colpali_query(query)

        prefetch = [
            models.Prefetch(
                query=image_embedding,
                using="dense",
                limit=5,
                # , score_threshold=0.5)
            )
        ]

        results = self.qdrant_client.query_points(
            collection_name=self.config.collection_name,
            prefetch=prefetch,
            query=colpali_embedding,
            using="mean_pooling",
            with_payload=True,
            limit=2,
        )

        logger.info(f"Retrieved points: {[point.id for point in results.points]}")
        return results.points

    def encode_image_to_base64(self, image_path):
        """Helper function to encode image to base64"""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                return encoded_string
        except Exception as e:
            logger.warning(f"Failed to encode image {image_path}: {e}")
            return None

    def index_documents(self, image_paths):
        points = []
        name_hashes = {}

        for i in tqdm(
            range(0, len(image_paths), 2)
        ):  # embedding functions process two images at once
            batch_paths = image_paths[i : i + 2]

            jina_embeddings = self.get_image_embeddings(image_paths=batch_paths)
            colpali_embeddings = self.get_colpali_embeddings(batch_paths)

            for path, jina_emb, colpali_emb in zip(
                batch_paths, jina_embeddings, colpali_embeddings
            ):
                stem = Path(path).stem
                base_name, page_str = stem.rsplit("_page_", 1)
                page_num = int(page_str)

                if base_name not in name_hashes:
                    name_hashes[base_name] = zlib.crc32(base_name.encode("utf-8")) % 10000000
                unique_id = name_hashes[base_name] + page_num

                base64_image = self.encode_image_to_base64(path)

                point = models.PointStruct(
                    id=unique_id,
                    vector={"dense": jina_emb, "mean_pooling": colpali_emb},
                    payload={
                        "image_path": str(path),
                        "filename": Path(path).name,
                        "base64": base64_image,
                    },
                )
                points.append(point)

                if i % 50 == 0:
                    self.qdrant_client.upsert(
                        collection_name=self.config.collection_name, points=points
                    )
                    points = []
                    logger.info(f"Upsert in progress - {i} documents completed")

        # upload any remaining points
        if points:
            self.qdrant_client.upsert(collection_name=self.config.collection_name, points=points)

        logger.info(f"Indexed {len(image_paths)} documents")
