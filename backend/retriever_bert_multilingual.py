import os
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from sentence_transformers import SentenceTransformer

DOCUMENTS_PATH = os.path.join(os.path.dirname(__file__), '../data/documents')
VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), '../data/vector_store')
COLLECTION_NAME = 'docs_bert_multilingual'

class BertMultilingualRetriever:
    def __init__(self, model_name: str = "bert-base-multilingual-cased", qdrant_url: str = None):
        self.model = SentenceTransformer(model_name)
        self.client = QdrantClient(path=VECTOR_STORE_PATH)
        self._init_collection()

    def _init_collection(self):
        if COLLECTION_NAME not in [c.name for c in self.client.get_collections().collections]:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )

    def embed_text(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

    def index_documents(self):
        doc_files = [f for f in os.listdir(DOCUMENTS_PATH) if f.endswith('.txt')]
        points = []
        for idx, fname in enumerate(doc_files):
            with open(os.path.join(DOCUMENTS_PATH, fname), 'r', encoding='utf-8') as f:
                content = f.read()
            embedding = self.embed_text(content)
            points.append(PointStruct(id=idx, vector=embedding, payload={"filename": fname, "content": content}))
        if points:
            self.client.upsert(collection_name=COLLECTION_NAME, points=points)

    def search(self, query: str, top_k: int = 3):
        query_vec = self.embed_text(query)
        hits = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=top_k
        )
        return [hit.payload for hit in hits]

    def close(self):
        del self.client 