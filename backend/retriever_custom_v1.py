import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import os

class RetrieverCustomV1:
    def __init__(self, model_name="lunara-kim/custom-embedding-slang-model"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.index = []  # (embedding, doc) 쌍 리스트
        self.documents = []

    def embed(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # [CLS] 토큰의 임베딩 사용 (BERT 계열)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return embedding

    def index_documents(self, documents_dir="data/documents"):
        self.index = []
        self.documents = []
        for fname in os.listdir(documents_dir):
            fpath = os.path.join(documents_dir, fname)
            if not fname.endswith(".txt"):
                continue
            with open(fpath, encoding="utf-8") as f:
                content = f.read()
            emb = self.embed(content)
            self.index.append(emb)
            self.documents.append({"content": content, "filename": fname})

    def search(self, query, top_k=3):
        if not self.index:
            self.index_documents()
        query_emb = self.embed(query)
        # 코사인 유사도 계산
        sims = [np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)) for doc_emb in self.index]
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [self.documents[i] for i in top_indices]

    def close(self):
        pass  # 리소스 정리 필요시 구현 