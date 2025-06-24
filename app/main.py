import streamlit as st
import os
from dotenv import load_dotenv
from backend.retriever_openai import OpenAIRetriever
from backend.retriever_sbert import SBERTRetriever
from backend.retriever_minicoil import MiniCOILRetriever
from backend.retriever_custom import CustomEmbeddingRetriever
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

# 임베딩 모델 선택 옵션에 Custom 추가
model_option = st.sidebar.selectbox("임베딩 모델 선택", ["OpenAI", "SBERT", "miniCOIL", "Custom"])

openai_api_key = os.environ.get("OPENAI_API_KEY", "")

# 모델 변경 시 초기화
# 모델 변경 시 초기화
if 'previous_model' not in st.session_state or st.session_state.previous_model != model_option:
    # 기존 retriever 안전하게 종료
    if 'retriever' in st.session_state and st.session_state.retriever:
        try:
            st.session_state.retriever.close()
        except Exception:
            pass
        del st.session_state.retriever
    st.session_state.model_initialized = False
    st.session_state.previous_model = model_option

# 모델별 초기화
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

elif model_option == "Custom":  # ✅ 새 모델용 분기
    if not openai_api_key:
        st.warning("OPENAI_API_KEY 환경변수를 설정하세요.")
        st.stop()
    if not st.session_state.model_initialized:
        with st.spinner("Custom 임베딩 모델 초기화 중..."):
            st.session_state.retriever = CustomEmbeddingRetriever()
            st.session_state.rag_chain = SimpleRAGChain(openai_api_key)
            st.session_state.model_initialized = True
        st.success("Custom 임베딩 모델이 초기화되었습니다!")

# 문서 인덱싱 버튼
if st.session_state.model_initialized and st.sidebar.button("문서 임베딩/인덱싱 갱신"):
    with st.spinner("문서를 임베딩하고 인덱싱 중입니다..."):
        st.session_state.retriever.index_documents()
    st.success("문서 인덱싱 완료!")

# 챗봇 UI
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
