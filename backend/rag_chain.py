import os
from openai import OpenAI
from typing import List

class SimpleRAGChain:
    def __init__(self, openai_api_key: str):
        self.llm = OpenAI(api_key=openai_api_key)

    def generate_answer(self, query: str, retrieved_docs: List[dict]) -> str:
        context = "\n\n".join([doc["content"] for doc in retrieved_docs])
        prompt = f"""다음 문서들을 참고하여 사용자의 질문에 답변하세요.\n\n[문서]\n{context}\n\n[질문]\n{query}\n\n[답변]"""
        print(prompt)
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512
        )
        return response.choices[0].message.content.strip() 