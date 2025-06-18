import streamlit as st
import os
from dotenv import load_dotenv
from backend.retriever_openai import OpenAIRetriever
from backend.retriever_sbert import SBERTRetriever
from backend.retriever_minicoil import MiniCOILRetriever
from backend.rag_chain import SimpleRAGChain

load_dotenv()

st.set_page_config(page_title="Custom Embedding RAG QA", layout="wide")
st.title("🔎 Custom Embedding 기반 RAG 챗봇")

# 세션 상태 초기화
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'model_initialized' not in st.session_state:
    st.session_state.model_initialized = False

# 임베딩 모델 선택
model_option = st.sidebar.selectbox("임베딩 모델 선택", ["OpenAI", "SBERT", "miniCOIL"])

openai_api_key = os.environ.get("OPENAI_API_KEY", "")

# 모델이 변경되었을 때 초기화 상태 리셋
if 'previous_model' not in st.session_state or st.session_state.previous_model != model_option:
    st.session_state.model_initialized = False
    st.session_state.previous_model = model_option

if model_option == "OpenAI":
    if not openai_api_key:
        st.warning("OPENAI_API_KEY 환경변수를 설정하세요.")
        st.stop()
    if not st.session_state.model_initialized:
        st.session_state.retriever = OpenAIRetriever(openai_api_key)
        st.session_state.rag_chain = SimpleRAGChain(openai_api_key)
        st.session_state.model_initialized = True
elif model_option == "SBERT":
    if not openai_api_key:
        st.warning("OPENAI_API_KEY 환경변수를 설정하세요.")
        st.stop()
    if not st.session_state.model_initialized:
        st.session_state.retriever = SBERTRetriever()
        st.session_state.rag_chain = SimpleRAGChain(openai_api_key)
        st.session_state.model_initialized = True
elif model_option == "miniCOIL":
    if not st.session_state.model_initialized:
        st.info("miniCOIL 모델을 사용하려면 아래 '모델 초기화' 버튼을 클릭하세요.")
        if st.button("모델 초기화"):
            with st.spinner("miniCOIL 모델을 초기화하는 중..."):
                st.session_state.retriever = MiniCOILRetriever()
                st.session_state.rag_chain = SimpleRAGChain(openai_api_key)
                st.session_state.model_initialized = True
            st.success("모델 초기화가 완료되었습니다!")

# 문서 인덱싱 버튼 (모델이 초기화된 경우에만 표시)
if st.session_state.model_initialized and st.sidebar.button("문서 임베딩/인덱싱 갱신"):
    with st.spinner("문서를 임베딩하고 인덱싱 중입니다..."):
        st.session_state.retriever.index_documents()
    st.success("문서 인덱싱 완료!")

# 챗봇 UI (모델이 초기화된 경우에만 표시)
if st.session_state.model_initialized:
    query = st.text_input("질문을 입력하세요:")

    if st.button("질문하기") and query:
        with st.spinner("검색 및 답변 생성 중..."):
            docs = st.session_state.retriever.search(query, top_k=3)
            answer = st.session_state.rag_chain.generate_answer(query, docs)
        st.markdown(f"#### 💬 답변\n{answer}")
        with st.expander("🔍 검색된 문서 보기"):
            for i, doc in enumerate(docs):
                st.markdown(f"**문서 {i+1}**\n\n{doc['content'][:300]}...") 