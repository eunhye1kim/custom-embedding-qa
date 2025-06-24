# custom-embedding-qa (한국어)

> 커스텀 임베딩 기반 의미 유사 질의응답 QA 시스템
> OpenAI, HuggingFace 임베딩 모델 비교 및 파인튜닝 실험 포함

## 📚 Other Languages

📚 [영문 README 보기](README.md)

---

## 📌 개요

이 프로젝트는 RAG(Retrieval-Augmented Generation) 시스템에서 발생하는 표현 다양성 및 문서 중복 문제를 커스텀 임베딩으로 해결하는 의미 기반 QA 시스템을 개발합니다.

Streamlit 기반 UI와 LangChain, Qdrant, OpenAI/HuggingFace 임베딩 모델을 활용하여 다양한 의미 검색 실험을 지원합니다.

---

## 🚀 주요 기능

* OpenAI, SBERT, miniCOIL, Custom 임베딩 모델 기반 QA 시스템
* Streamlit 챗봇 인터페이스 (app/main.py)
* 사이드바에서 임베딩 모델 선택 (OpenAI, SBERT, miniCOIL, Custom)
* 문서 임베딩/인덱싱 갱신 버튼
* 임베딩 모델 파인튜닝 및 서빙 (finetune/)
* 검색 성능 비교 (NDCG, 정성 평가 등)

---

## 💻 기술 스택

| 분류     | 도구                              |
| ------ | ------------------------------- |
| 언어     | Python                          |
| 프레임워크  | LangChain, Streamlit            |
| 임베딩 모델 | OpenAI, MiniCoIL, SBERT, Custom |
| 벡터 DB  | Qdrant                          |
| 개발 환경  | Colab, VSCode, Jupyter Notebook |

---

## 📁 프로젝트 구조

```
custom-embedding-qa/
├── app/             # Streamlit UI (main.py)
├── backend/         # 임베딩 및 검색 로직 (리트리버, RAG 체인)
├── data/            # 문서 데이터 및 벡터 DB
├── finetune/        # 모델 파인튜닝 및 서빙
├── config/          # 설정 파일
├── requirements.txt
├── README.md        # 영문 버전
└── README.ko.md     # 한글 버전
```

---

## 🚦 실행 방법

1. (선택) 가상환경 생성 및 활성화

**Windows (CMD):**
```bash
python -m venv env
env\Scripts\activate
```
**Windows (PowerShell):**
```bash
python -m venv env
.\env\Scripts\Activate.ps1
```
**macOS/Linux:**
```bash
python3 -m venv env
source env/bin/activate
```

2. 필수 패키지 설치

```bash
pip install -r requirements.txt
```

3. 환경 변수 설정
   루트 디렉토리에 `.env` 파일을 생성하고 API 키(예: `OPENAI_API_KEY=...`)를 입력하세요.

4. Streamlit 앱 실행 (모듈 임포트 위해 PYTHONPATH 설정 필요)

**Windows (CMD):**
```bash
set PYTHONPATH=.
streamlit run app/main.py
```
**Windows (PowerShell):**
```bash
$env:PYTHONPATH="."
streamlit run app/main.py
```
**macOS/Linux:**
```bash
PYTHONPATH=. streamlit run app/main.py
```

5. (선택) 가상환경 비활성화
```bash
deactivate
```

---

## 🧩 동작 방식

- **모델 선택:** 사이드바에서 임베딩 모델(OpenAI, SBERT, miniCOIL, Custom) 선택
- **문서 인덱싱:** 사이드바 버튼으로 임베딩/인덱싱 갱신
- **챗봇 UI:** 질문 입력, 답변 및 검색된 문서 확인
- **백엔드:** backend/ 내 모듈형 리트리버 및 RAG 체인
- **파인튜닝:** finetune/ 내 커스텀 모델 학습/서빙 스크립트

---

## 👩‍💻 작성자

김은혜
Machine Learning Engineer / AI 시스템 빌더
목표: 커스텀 임베딩 기반 의미 검색 RAG 구조 실험 및 포트폴리오 정리
