import streamlit as st
import os
from dotenv import load_dotenv
from backend.retriever_openai import OpenAIRetriever
from backend.retriever_sbert import SBERTRetriever
from backend.rag_chain import SimpleRAGChain

load_dotenv()

st.set_page_config(page_title="Custom Embedding RAG QA", layout="wide")
st.title("🔎 Custom Embedding 기반 RAG 챗봇")

# 임베딩 모델 선택
model_option = st.sidebar.selectbox("임베딩 모델 선택", ["OpenAI", "SBERT"])

openai_api_key = os.environ.get("OPENAI_API_KEY", "")

if model_option == "OpenAI":
    if not openai_api_key:
        st.warning("OPENAI_API_KEY 환경변수를 설정하세요.")
        st.stop()
    retriever = OpenAIRetriever(openai_api_key)
    rag_chain = SimpleRAGChain(openai_api_key)
elif model_option == "SBERT":
    if not openai_api_key:
        st.warning("OPENAI_API_KEY 환경변수를 설정하세요.")
        st.stop()
    retriever = SBERTRetriever()
    rag_chain = SimpleRAGChain(openai_api_key)  # 답변 생성은 여전히 OpenAI GPT 사용

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