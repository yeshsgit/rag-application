from sentence_transformers import SentenceTransformer
import chromadb
from progressbar import progressbar


class embedder():

    def __init__(self, raw_chunk_list: list[dict], embedding_model: str) -> None:
        self.raw_chunk_list = raw_chunk_list
        self.embedding_model = embedding_model
        self.encoder = SentenceTransformer(self.embedding_model, trust_remote_code=True)
        self.client = chromadb.PersistentClient()

        self.collection = self.client.get_or_create_collection(name="chunks")
        if self.collection.count() == len(self.raw_chunk_list):
            print("[INFO] Vector database with embeddings already exist")
            self.collection = self.client.get_collection("chunks")
        else:
            print(f"[INFO] Embedding chunks with model {self.encoder.model_card_data.base_model} and saving to vector database")
            self.client.delete_collection(name="chunks")
            self.collection = self.client.create_collection("chunks")
            embeddings = self.encoder.encode(self.raw_chunk_list, show_progress_bar=True)
            print("[INFO] Storing embeddings in vectordb")
            for i, e in progressbar(enumerate(embeddings), max_value=len(embeddings)):
                self.collection.add(
                    ids=[str(i)],
                    embeddings=[e.tolist()],
                    documents=self.raw_chunk_list[i]
                )

    def embed(self, prompt: str | None = None) -> chromadb.Collection | list:
        """Returns embeddings"""
        if prompt:
            embedding = self.encoder.encode(prompt)
            return embedding.tolist()
        else:
            return self.collection
