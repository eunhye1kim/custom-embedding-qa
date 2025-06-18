import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    SparseVectorParams,
    Distance,
    SparseIndexParams,
    SparseVector,
    NamedSparseVector
)
from fastembed import SparseTextEmbedding
import numpy as np
import streamlit as st

DOCUMENTS_PATH = os.path.join(os.path.dirname(__file__), '../data/documents')
COLLECTION_NAME = 'docs_minicoil'

class MiniCOILRetriever:
    def __init__(self, model_name: str = "Qdrant/minicoil-v1"):
        self.model_name = model_name
        self.model = None
        
        # 메모리 전용 모드로 Qdrant 클라이언트 초기화
        if 'qdrant_client' not in st.session_state:
            st.session_state.qdrant_client = QdrantClient(":memory:")
        self.client = st.session_state.qdrant_client
        
        self._init_collection()
        
    def _init_model(self):
        """모델을 실제로 필요할 때 초기화"""
        if self.model is None:
            self.model = SparseTextEmbedding(model_name=self.model_name)

    def _init_collection(self):
        """Qdrant 컬렉션 초기화"""
        collections = self.client.get_collections().collections
        if not any(collection.name == COLLECTION_NAME for collection in collections):
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                sparse_vectors_config={
                    "text": SparseVectorParams(
                        index=SparseIndexParams(
                            on_disk=False
                        )
                    )
                }
            )

    def _process_sparse_vector(self, text: str) -> tuple[list[int], list[float]]:
        """텍스트를 sparse 벡터로 변환"""
        self._init_model()
        try:
            # generator 객체를 받아서 첫 번째 값만 사용
            embedding_gen = self.model.embed(text)
            embedding = next(embedding_gen)
            
            # SparseT 객체 처리
            if hasattr(embedding, 'indices') and hasattr(embedding, 'values'):
                indices = list(embedding.indices)
                values = list(embedding.values)
            elif isinstance(embedding, tuple):
                indices, values = embedding
                indices = list(indices)
                values = list(values)
            else:
                # numpy array 처리
                embedding = np.atleast_1d(embedding)
                nonzero_mask = embedding != 0
                indices = np.where(nonzero_mask)[0].tolist()
                values = embedding[nonzero_mask].tolist()
            
            return indices, values
        except Exception as e:
            return [], []

    def index_documents(self, documents: List[str]) -> None:
        """문서 인덱싱"""
        points = []
        for idx, doc in enumerate(documents):
            indices, values = self._process_sparse_vector(doc)
            points.append(
                PointStruct(
                    id=idx,
                    payload={"text": doc},
                    sparse_vectors={
                        "text": SparseVector(
                            indices=indices,
                            values=values
                        )
                    }
                )
            )
        
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """검색 수행"""
        indices, values = self._process_sparse_vector(query)
        
        # 검색 결과가 없는 경우 처리
        if not indices or not values:
            return []
            
        sparse_vector = NamedSparseVector(
            name="text",
            vector=SparseVector(
                indices=indices,
                values=values
            )
        )
            
        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=sparse_vector,
            limit=top_k
        )
        
        return [
            {
                "text": result.payload["text"],
                "score": result.score
            }
            for result in results
        ] 