import streamlit as st
import os
from backend.retriever_openai import OpenAIRetriever
from backend.rag_chain import SimpleRAGChain

st.set_page_config(page_title="Custom Embedding RAG QA", layout="wide")
st.title("🔎 Custom Embedding 기반 RAG 챗봇")

# API 키 입력
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if not openai_api_key:
    st.warning("좌측 사이드바에 OpenAI API Key를 입력하세요.")
    st.stop()

retriever = OpenAIRetriever(openai_api_key)
rag_chain = SimpleRAGChain(openai_api_key)

# 문서 인덱싱 버튼
if st.sidebar.button("문서 임베딩/인덱싱 갱신"):
    with st.spinner("문서를 임베딩하고 인덱싱 중입니다..."):
        retriever.index_documents()
    st.success("문서 인덱싱 완료!")

# 챗봇 UI
query = st.text_input("질문을 입력하세요:")

if st.button("질문하기") and query:
    with st.spinner("검색 및 답변 생성 중..."):
        docs = retriever.search(query, top_k=3)
        answer = rag_chain.generate_answer(query, docs)
    st.markdown(f"#### 💬 답변\n{answer}")
    with st.expander("🔍 검색된 문서 보기"):
        for i, doc in enumerate(docs):
            st.markdown(f"**문서 {i+1}: {doc['filename']}**\n\n{doc['content'][:300]}...") 